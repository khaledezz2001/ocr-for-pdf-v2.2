import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

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
        # Attention optimization for H200
        attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
    )

    model.eval()
    
    # Compile model for faster inference (PyTorch 2.x)
    if hasattr(torch, 'compile'):
        log("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    # Warmup inference
    log("Running warmup inference...")
    warmup_image = Image.new('RGB', (1600, 1200), color='white')
    _ = ocr_image(warmup_image)
    
    # Clear cache after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log("RolmOCR model loaded and optimized")


# ===============================
# OCR IMAGE (OPTIMIZED)
# ===============================
def ocr_image(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Return the COMPLETE plain text of this document from top to bottom, "
                        "including headers, tables, footers, bank details, signatures, stamps, "
                        "emails, phone numbers, and all numbers exactly as written."
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
        return_tensors="pt"
    ).to(DEVICE, dtype=torch.bfloat16)  # Ensure BF16 for inputs

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1200,
            temperature=0.1,
            do_sample=False,  # Greedy decoding for speed
            num_beams=1,      # No beam search for faster generation
            use_cache=True,   # Enable KV cache
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    # Clean output
    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1]

    decoded = decoded.strip()
    while decoded.startswith(("system\n", "user\n", ".")):
        decoded = decoded.split("\n", 1)[-1].strip()

    decoded = decoded.replace("assistant\n", "", 1).strip()

    return decoded


# ===============================
# HANDLER (IMAGE ONLY → TEXT ONLY)
# ===============================
def handler(event):
    load_model()

    if "image" not in event["input"]:
        return {
            "status": "error",
            "message": "Only image input is supported"
        }

    try:
        image = decode_image(event["input"]["image"])
        text = ocr_image(image)

        # Optional: Clear CUDA cache after each request to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "text": text
        }
    except Exception as e:
        log(f"Error processing request: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
# PRELOAD
# ===============================
log("Preloading model...")
load_model()

runpod.serverless.start({
    "handler": handler
})