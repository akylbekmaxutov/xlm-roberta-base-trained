import os
import io
import pickle
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# OCR
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ======================
# Config
# ======================
# If you uploaded your fine-tuned model to the Hub:
MODEL_REPO = "akylbekmaxutov/xlm-roberta-base-trained"   # <-- change to your repo id
# Or load from a local folder checked into the Space:
# MODEL_REPO = "./xlmr_multiclass_ministry_final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
STRIDE = 64
TOPK = 5

# Tesseract OCR language(s). "rus" for Russian; add "eng" if mixed: "rus+eng"
TESS_LANG = os.getenv("TESS_LANG", "rus")

# If your model is private, add a Space secret HF_TOKEN and enable login:
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    from huggingface_hub import login
    login(HF_TOKEN)


# ======================
# Model / tokenizer / labels
# ======================
def load_label_encoder(repo_or_path: str):
    # Works both for local folder and HF repo clone in cache (Spaces pulls files to working dir)
    pkl_candidates = [
        os.path.join(repo_or_path, "label_encoder.pkl"),
        "./label_encoder.pkl",
    ]
    for p in pkl_candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("label_encoder.pkl not found. Put it in your model repo/folder.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, local_files_only=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).to(DEVICE)
model.eval()

label_encoder = load_label_encoder(MODEL_REPO)
id2label = {i: str(lbl) for i, lbl in enumerate(label_encoder.classes_)}


# ======================
# PDF -> text (native)
# ======================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text using PDF text layer (works for digital PDFs, not scans).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    for page in doc:
        chunks.append(page.get_text("text"))
    return "\n".join(chunks).strip()


# ======================
# OCR fallback (scans/handwritten)
# ======================
def ocr_pdf(file_bytes: bytes, dpi: int = 300) -> str:
    """
    Converts each page to an image and runs Tesseract OCR.
    """
    images = convert_from_bytes(file_bytes, dpi=dpi)  # uses poppler
    texts = []
    for img in images:
        # You can pre-process img for better OCR (e.g., grayscale, threshold)
        text = pytesseract.image_to_string(img, lang=TESS_LANG)
        texts.append(text)
    return "\n".join(texts).strip()


# ======================
# Token chunking (long docs)
# ======================
def chunk_tokens(text: str, max_length=MAX_LENGTH, stride=STRIDE):
    enc = tokenizer(text, truncation=False, padding=False, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]

    start = 0
    while start < len(input_ids):
        end = start + max_length
        ids_chunk = input_ids[start:end]
        mask_chunk = attn[start:end]

        if len(ids_chunk) < max_length:
            pad_len = max_length - len(ids_chunk)
            pad_id = tokenizer.pad_token_id or 1
            ids_chunk = torch.cat([ids_chunk, torch.full((pad_len,), pad_id)])
            mask_chunk = torch.cat([mask_chunk, torch.zeros(pad_len, dtype=torch.long)])

        yield {
            "input_ids": ids_chunk.unsqueeze(0),
            "attention_mask": mask_chunk.unsqueeze(0),
        }

        if end >= len(input_ids):
            break
        start = end - stride


# ======================
# Inference
# ======================
def predict_topk_from_text(text: str, topk=TOPK):
    text = (text or "").strip()
    if not text:
        return {}, []

    logits_sum = None
    with torch.no_grad():
        for ch in chunk_tokens(text):
            input_ids = ch["input_ids"].to(DEVICE)
            attention_mask = ch["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, C]
            logits_sum = out if logits_sum is None else (logits_sum + out)

    probs = F.softmax(logits_sum, dim=-1).squeeze(0)  # [C]
    top_probs, top_ids = torch.topk(probs, k=min(topk, probs.size(0)))
    top_probs = top_probs.cpu().tolist()
    top_ids = top_ids.cpu().tolist()

    label_probs = {id2label[i]: float(p) for i, p in zip(top_ids, top_probs)}
    table = [{"Department": id2label[i], "Probability": round(float(p) * 100, 2)} for i, p in zip(top_ids, top_probs)]
    return label_probs, table


def predict_from_pdf(file: gr.File, force_ocr: bool):
    if file is None:
        return {}, [], "(no file)", "(no text)"
    if file.size and file.size > 20 * 1024 * 1024:
        return {}, [], "error", "PDF too large (>20MB)."

    with open(file.name, "rb") as f:
        content = f.read()

    native_text = ""
    ocr_text = ""
    used = "native"

    try:
        if not force_ocr:
            native_text = extract_text_from_pdf(content)
    except Exception as e:
        native_text = ""
    if force_ocr or not native_text:
        try:
            ocr_text = ocr_pdf(content, dpi=300)
            used = "ocr"
        except Exception as e:
            ocr_text = ""

    text = native_text if (used == "native") else ocr_text
    if not text:
        return {}, [], used, "No text found (native & OCR failed)."

    label_probs, table = predict_topk_from_text(text, TOPK)
    # Preview just the first 1500 chars
    preview = text[:1500] + ("..." if len(text) > 1500 else "")
    return label_probs, table, used, preview


# ======================
# Gradio UI
# ======================
with gr.Blocks(title="Department Classifier (PDF + OCR)") as demo:
    gr.Markdown("# Department Classifier")
    gr.Markdown(
        "Upload a **PDF** in Russian. We extract text. If there’s no text layer (scanned/handwritten), "
        "we’ll run **OCR** (Tesseract) and then classify with your fine-tuned XLM-R model."
    )

    with gr.Row():
        pdf_input = gr.File(label="PDF file", file_types=[".pdf"])
    with gr.Row():
        force_ocr = gr.Checkbox(label="Force OCR (skip native text extraction)", value=False)

    with gr.Row():
        topk_label = gr.Label(label="Top-5 Departments (probabilities)", num_top_classes=5)
    topk_table = gr.Dataframe(headers=["Department", "Probability"], interactive=False)

    with gr.Accordion("Debug / Preview", open=False):
        method = gr.Textbox(label="Extraction method used", interactive=False)
        preview = gr.Textbox(label="Extracted text (preview)", lines=10, interactive=False)

    btn = gr.Button("Predict")
    btn.click(fn=predict_from_pdf, inputs=[pdf_input, force_ocr],
              outputs=[topk_label, topk_table, method, preview])

if __name__ == "__main__":
    # On Spaces, just running app.py is fine
    demo.launch(server_name="0.0.0.0", server_port=7860)
