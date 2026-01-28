import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PAGES = 30

processor = None
model = None

# ===============================
# H200 OPTIMIZATIONS
# ===============================
if torch.cuda.is_available():
    # Enable TF32 for faster matmul on Ampere/Hopper GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for optimal kernels
    torch.backends.cudnn.benchmark = True
    
    # Disable deterministic mode for speed
    torch.backends.cudnn.deterministic = False
    
    # Set optimal memory allocator settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


# ===============================
# HALLUCINATION DETECTION
# ===============================
def is_hallucinated_output(text: str) -> bool:
    """Detect if the OCR output is hallucinated/garbage"""
    if not text or len(text.strip()) < 10:
        return True
    
    text_lower = text.lower()
    
    # Common hallucination phrases that models generate for empty pages
    hallucination_indicators = [
        "table 1:",
        "comparison of different methods",
        "note: the choice of method",
        "this page is blank",
        "no text found",
        "empty page",
        "the image appears to be",
        "there is no visible text",
        "the document appears to be blank",
        "i cannot see any text",
        "method | accuracy | speed",
        "soil moisture",
        "time domain reflectometry"
    ]
    
    # Check if text contains hallucination phrases
    for indicator in hallucination_indicators:
        if indicator in text_lower:
            return True
    
    # Check for repetitive table patterns
    lines = text.strip().split('\n')
    if len(lines) > 20:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < 3:
            return True
    
    # Check for excessive markdown tables (generic hallucinations)
    table_markers = text.count('|')
    pipe_lines = sum(1 for line in lines if '|' in line)
    
    # If more than 50% of lines have pipes, likely a hallucinated table
    if len(lines) > 0 and pipe_lines / len(lines) > 0.5:
        # Check if it's a real table with actual content or generic hallucination
        content_without_pipes = text.replace('|', '').replace('-', '').replace('\n', '').strip()
        if len(content_without_pipes) < 100:  # Too little actual content
            return True
    
    # Check for suspiciously perfect table formatting (hallucination signature)
    if table_markers > 10:
        # Real tables usually have irregular content
        # Hallucinated tables often have very uniform structure
        table_rows = [line for line in lines if '|' in line]
        if len(table_rows) > 3:
            # Count pipes per row
            pipe_counts = [line.count('|') for line in table_rows]
            # If all rows have exactly the same number of pipes, suspicious
            if len(set(pipe_counts)) == 1 and pipe_counts[0] > 3:
                return True
    
    # Check for only special characters
    alphanumeric_chars = sum(c.isalnum() for c in text)
    if alphanumeric_chars < 10:
        return True
    
    return False


# ===============================
# IMAGE DECODING (OPTIMIZED)
# ===============================
def decode_image(b64):
    """Optimized image decoding with faster resampling"""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    target_width = 1600
    scale = target_width / img.width
    
    # Use LANCZOS only if downscaling significantly, otherwise BILINEAR is faster
    if scale < 0.5:
        resample = Image.LANCZOS
    else:
        resample = Image.BILINEAR
    
    img = img.resize(
        (target_width, int(img.height * scale)),
        resample
    )
    return img


def decode_pdf(b64):
    """Decode PDF to images"""
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes,
        dpi=150,
        fmt="png",
        thread_count=4,
        use_pdftocairo=True
    )
    return images[:MAX_PAGES]


# ===============================
# LOAD MODEL ONCE (OPTIMIZED)
# ===============================
def load_model():
    global processor, model
    if model is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # BF16 for better speed on H200
        local_files_only=True,
        low_cpu_mem_usage=True,
        # Attention optimization for H200
        attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
    )

    model.eval()
    
    # Compile model for faster inference (PyTorch 2.x)
    if hasattr(torch, 'compile'):
        log("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    log("RolmOCR model loaded and optimized")


# ===============================
# OCR ONE PAGE (OPTIMIZED)
# ===============================
def ocr_page(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are a professional OCR system. Extract ALL text from this document "
                        "EXACTLY as written. Include:\n"
                        "- All headers, titles, and sections\n"
                        "- All body text and paragraphs\n"
                        "- All tables with correct alignment\n"
                        "- All numbers, dates, and codes EXACTLY as shown\n"
                        "- All names, addresses, and contact information\n"
                        "- All signatures, stamps, and annotations\n"
                        "- Preserve original spelling and formatting\n\n"
                        "CRITICAL RULES:\n"
                        "- Do NOT correct typos or translate anything\n"
                        "- Do NOT add interpretations or summaries\n"
                        "- Do NOT make up content if the page is blank or empty\n"
                        "- If the page is truly empty, output only: EMPTY_PAGE\n"
                        "- Do NOT create tables, examples, or sample data\n\n"
                        "Return ONLY the extracted text, nothing else."
                    )
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(DEVICE, dtype=torch.bfloat16, non_blocking=True)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1536,
            min_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Clean up response
    if "assistant" in decoded.lower():
        idx = decoded.lower().index("assistant") + len("assistant")
        decoded = decoded[idx:]

    return decoded.strip()


# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()

    # Prefix to remove from output
    PREFIX = ".\nuser\nYou are a professional OCR system. Extract ALL text from this document EXACTLY as written. Include:\n- All headers, titles, and sections\n- All body text and paragraphs\n- All tables with correct alignment\n- All numbers, dates, and codes EXACTLY as shown\n- All names, addresses, and contact information\n- All signatures, stamps, and annotations\n- Preserve original spelling and formatting\n\nCRITICAL RULES:\n- Do NOT correct typos or translate anything\n- Do NOT add interpretations or summaries\n- Do NOT make up content if the page is blank or empty\n- If the page is truly empty, output only: EMPTY_PAGE\n- Do NOT create tables, examples, or sample data\n\nReturn ONLY the extracted text, nothing else.\nassistant\n"

    try:
        if "image" in event["input"]:
            pages = [decode_image(event["input"]["image"])]
        elif "file" in event["input"]:
            pages = decode_pdf(event["input"]["file"])
        else:
            return {
                "status": "error",
                "message": "Missing image or file"
            }

        extracted_pages = []

        for i, page in enumerate(pages, start=1):
            text = ocr_page(page)
            
            # Remove prefix
            text = text.replace(PREFIX, "", 1).strip()
            
            # Check if model explicitly said it's empty
            if text.upper() == "EMPTY_PAGE" or text.upper().startswith("EMPTY_PAGE"):
                text = "[Empty or unreadable page]"
            # Detect hallucinations
            elif is_hallucinated_output(text):
                log(f"Warning: Page {i} appears to be hallucinated")
                text = "[Empty or unreadable page]"
            
            extracted_pages.append({
                "page": i,
                "text": text
            })
            
            # Clear cache after each page to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages
        }

    except Exception as e:
        log(f"Error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
# PRELOAD & WARMUP
# ===============================
log("Preloading model...")
load_model()

# Warmup with smaller image
if torch.cuda.is_available():
    log("Running warmup...")
    dummy_image = Image.new('RGB', (1600, 1200), color='white')
    try:
        _ = ocr_page(dummy_image)
        torch.cuda.empty_cache()
        log("Warmup complete")
    except Exception as e:
        log(f"Warmup failed: {e}")

runpod.serverless.start({
    "handler": handler
})
