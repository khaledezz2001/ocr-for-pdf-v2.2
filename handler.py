import os
import base64
import io
import time

import torch
import runpod
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb guard for large PDFs
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/allenai/olmOCR-2-7B-1025-FP8"
MAX_PAGES = 100
MAX_NEW_TOKENS = 1536
BATCH_SIZE = 8   # pages per forward pass — tune down if OOM, up for speed

model = None
processor = None

def log(msg):
    print(f"[BOOT] {msg}", flush=True)

# ===============================
# HALLUCINATION DETECTION
# ===============================
def is_hallucinated_output(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True

    text_lower = text.lower()

    hallucination_indicators = [
        "table 1:", "comparison of different methods", "note: the choice of method",
        "this page is blank", "no text found", "empty page", "the image appears to be",
        "there is no visible text", "the document appears to be blank", "i cannot see any text",
        "method | accuracy | speed", "soil moisture", "time domain reflectometry"
    ]
    for indicator in hallucination_indicators:
        if indicator in text_lower:
            return True

    lines = text.strip().split('\n')
    if len(lines) > 20:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < 3:
            return True

    pipe_lines = sum(1 for line in lines if '|' in line)
    if len(lines) > 0 and pipe_lines / len(lines) > 0.5:
        content_without_pipes = text.replace('|', '').replace('-', '').replace('\n', '').strip()
        if len(content_without_pipes) < 100:
            return True

    table_markers = text.count('|')
    if table_markers > 10:
        table_rows = [line for line in lines if '|' in line]
        if len(table_rows) > 3:
            pipe_counts = [line.count('|') for line in table_rows]
            if len(set(pipe_counts)) == 1 and pipe_counts[0] > 3:
                return True

    if sum(c.isalnum() for c in text) < 10:
        return True

    return False

# ===============================
# IMAGE DECODING
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    target_width = 1600
    scale = target_width / img.width
    img = img.resize((target_width, int(img.height * scale)), Image.BICUBIC)
    return img

def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes, dpi=150, fmt="png", thread_count=4, use_pdftocairo=True,
        size=(1400, None),  # cap width to keep pixel count under bomb limit
    )
    # Resize oversized pages to keep memory under control
    resized = []
    for img in images[:MAX_PAGES]:
        img = img.convert("RGB")
        if img.width > 1600:
            scale = 1600 / img.width
            img = img.resize((1600, int(img.height * scale)), Image.BICUBIC)
        resized.append(img)
    return resized

# ===============================
# LOAD MODEL ONCE
# ===============================
def load_model():
    global model, processor
    if model is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

    log("Loading model onto GPU...")
    # Load as bfloat16 — the FP8 checkpoint is dequantized on load.
    # FP8 inference kernels are not available on Blackwell (sm_120) yet,
    # so running in FP8 falls back to slow scalar ops (~90s/page).
    # bfloat16 uses native tensor cores and is ~10x faster here.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        attn_implementation="sdpa",  # scaled dot-product attention — faster than eager
    )
    model.eval()

    log("Compiling model with torch.compile...")
    # torch.compile fuses ops and enables CUDA graphs — big win on Blackwell.
    # "reduce-overhead" is the best mode for repeated same-shape inference.
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    log("Model loaded and compiled successfully")

# ===============================
# OCR PROMPT
# ===============================
OCR_PROMPT_TEXT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML."
)

def build_messages(image: Image.Image) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": OCR_PROMPT_TEXT}
            ]
        }
    ]

# ===============================
# BATCH OCR (transformers)
# ===============================
def ocr_batch(images: list) -> list:
    results = []

    for i in range(0, len(images), BATCH_SIZE):
        chunk = images[i:i + BATCH_SIZE]

        # Build per-image chat messages and apply the chat template
        texts = []
        all_images = []
        for img in chunk:
            messages = build_messages(img)
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
            all_images.append(img)

        inputs = processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
            )

        # Decode only the newly generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        for gen_ids in generated_ids:
            new_tokens = gen_ids[input_len:]
            decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(decoded.strip())

        del inputs, generated_ids
        torch.cuda.empty_cache()

    return results

# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()

    try:
        if "image" in event["input"]:
            pages = [decode_image(event["input"]["image"])]
        elif "file" in event["input"]:
            pages = decode_pdf(event["input"]["file"])
        else:
            return {"status": "error", "message": "Missing image or file"}

        total_pages = len(pages)
        log(f"Processing {total_pages} pages using transformers...")
        start_time = time.time()

        batch_results = ocr_batch(pages)

        extracted_pages = []
        for j, text in enumerate(batch_results):
            page_num = j + 1

            if text.upper().startswith("EMPTY_PAGE"):
                text = "[Empty or unreadable page]"
            elif is_hallucinated_output(text):
                log(f"Warning: Page {page_num} appears to be hallucinated")
                text = "[Empty or unreadable page]"

            extracted_pages.append({"page": page_num, "text": text})

        torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        log(f"Completed {total_pages} pages in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages
        }

    except Exception as e:
        log(f"Error: {str(e)}")
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}

# ===============================
# PRELOAD & WARMUP
# ===============================
log("Preloading model...")
load_model()

if torch.cuda.is_available():
    log("Running dummy warmup...")
    dummy_image = Image.new('RGB', (1600, 1200), color='white')
    try:
        _ = ocr_batch([dummy_image])
        log("Warmup complete!")
    except Exception as e:
        log(f"Warmup error: {e}")

runpod.serverless.start({"handler": handler})
