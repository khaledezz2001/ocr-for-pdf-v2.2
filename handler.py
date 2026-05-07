import os
import base64
import io
import time
import runpod
from PIL import Image
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"

# ===============================
# CONFIG
# ===============================
MODEL_PATH      = "/models/hf/datalab-to/chandra-ocr-2"
MAX_PAGES       = 100
MAX_NEW_TOKENS  = 32768   # was 12384 — increased to prevent truncation on dense pages
MAX_NUM_SEQS    = 4       # reduced from 8 to free up memory for larger context

# ---------------------------------------------------------------
# IMAGE QUALITY SETTINGS
# ---------------------------------------------------------------
PDF_DPI         = 300
TARGET_WIDTH    = 2400

llm = None

def log(msg):
    print(f"[HANDLER] {msg}", flush=True)

# ===============================
# IMAGE HELPERS
# ===============================
def decode_image(b64: str) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    scale = TARGET_WIDTH / img.width
    return img.resize((TARGET_WIDTH, int(img.height * scale)), Image.BICUBIC)

def decode_pdf(b64: str) -> list:
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes,
        dpi=PDF_DPI,
        fmt="png",
        thread_count=4,
        use_pdftocairo=True,
    )
    resized = []
    for img in images[:MAX_PAGES]:
        scale = TARGET_WIDTH / img.width
        resized.append(
            img.resize((TARGET_WIDTH, int(img.height * scale)), Image.BICUBIC)
        )
    return resized

# ===============================
# LOAD vLLM IN-PROCESS (once)
# ===============================
def load_model():
    global llm
    if llm is not None:
        return

    from vllm import LLM

    log("Loading Chandra 2 via vLLM (in-process)...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=32768,          # was 8192 — must match MAX_NEW_TOKENS
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.92,  # slightly higher to accommodate larger context
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,
    )
    log("Chandra 2 vLLM engine ready")

# ===============================
# BUILD CHANDRA PROMPT
# ===============================
def build_prompt(img: Image.Image) -> dict:
    from chandra.prompts import PROMPT_MAPPING
    from transformers import AutoProcessor

    if not hasattr(build_prompt, "_processor"):
        build_prompt._processor = AutoProcessor.from_pretrained(
            MODEL_PATH, local_files_only=True
        )

    processor = build_prompt._processor
    ocr_prompt = PROMPT_MAPPING["ocr_layout"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": ocr_prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": img},
    }

# ===============================
# BATCH OCR via vLLM offline API
# ===============================
def ocr_batch(images: list) -> list:
    from vllm import SamplingParams
    from chandra.output import parse_markdown

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=0.1,
        repetition_penalty=1.05,
    )

    inputs = [build_prompt(img) for img in images]
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    results = []
    for out in outputs:
        raw = out.outputs[0].text.strip()

        # Check if output was truncated (didn't finish naturally)
        finish_reason = out.outputs[0].finish_reason
        if finish_reason == "length":
            log(f"WARNING: Output truncated — consider increasing MAX_NEW_TOKENS further")

        try:
            text = parse_markdown(raw)
        except Exception:
            text = raw
        results.append(text)

    return results

# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    load_model()
    try:
        inp = event.get("input", {})
        if "image" in inp:
            pages = [decode_image(inp["image"])]
        elif "file" in inp:
            pages = decode_pdf(inp["file"])
        else:
            return {"status": "error", "message": "Missing 'image' or 'file' in input"}

        total_pages = len(pages)
        log(f"Processing {total_pages} page(s) at {PDF_DPI} DPI / {TARGET_WIDTH}px width...")
        t0 = time.time()

        raw_results = ocr_batch(pages)

        extracted_pages = []
        for j, text in enumerate(raw_results):
            page_num = j + 1
            if not text or len(text.strip()) < 5:
                text = "[Empty or unreadable page]"
            extracted_pages.append({"page": page_num, "text": text})

        elapsed = time.time() - t0
        log(f"Done: {total_pages} pages in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages,
        }

    except Exception as e:
        import traceback
        log(f"Error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

# ===============================
# ENTRY POINT
# *** __main__ guard prevents vLLM spawn workers from re-running this ***
# ===============================
if __name__ == "__main__":
    log("Cold start — loading Chandra 2 via vLLM...")
    load_model()

    log("Running warmup pass...")
    try:
        dummy = Image.new("RGB", (TARGET_WIDTH, 1800), color="white")
        _ = ocr_batch([dummy])
        log("Warmup complete!")
    except Exception as e:
        log(f"Warmup error (non-fatal): {e}")

    runpod.serverless.start({"handler": handler})
