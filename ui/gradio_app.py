"""Gradio UI for the Multimodal RAG system."""

import os

import gradio as gr
import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:3001")
REQUEST_TIMEOUT = 120  # seconds (model inference can be slow)

# Demo limits (should match pipeline.py)
MAX_PAGES = 100
MAX_FILE_SIZE_MB = 50


def check_api_health() -> str:
    """Check if the API is reachable and get stats."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        pages = data.get('indexed_pages', 0)

        # Try to get more detailed stats
        try:
            stats_r = requests.get(f"{API_BASE}/stats", timeout=5)
            stats_r.raise_for_status()
            stats = stats_r.json()
            docs = stats.get('num_docs', 0)
            device = stats.get('device', 'cpu')
            gpu = stats.get('gpu_name', '')
            device_info = f"{device}" + (f" ({gpu})" if gpu else "")
            return f"✅ API connected | {docs} docs, {pages} pages indexed | Device: {device_info}"
        except Exception:
            return f"✅ API connected | {pages} pages indexed"
    except requests.RequestException:
        return "❌ API not reachable. Is the API server running?"


def ingest_pdf(pdf_file) -> str:
    """Upload and ingest a PDF document."""
    if pdf_file is None:
        return "⚠️ No file selected."

    # Client-side file size check
    try:
        file_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return f"❌ File too large: {file_size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit"
    except Exception:
        pass  # Proceed to server validation

    try:
        with open(pdf_file, "rb") as f:
            r = requests.post(
                f"{API_BASE}/ingest",
                files={"file": (os.path.basename(pdf_file), f, "application/pdf")},
                timeout=REQUEST_TIMEOUT,
            )

        if r.status_code == 400:
            # Graceful error from server (demo limits, validation)
            error_detail = r.json().get("detail", r.text)
            return f"⚠️ {error_detail}"

        r.raise_for_status()
        j = r.json()

        status = "📄 New document" if j.get("is_new") else "📄 Already indexed (no duplicates added)"
        return f"{status} | doc_id: {j['doc_id']} | pages: {j['num_pages']}"

    except requests.HTTPError as e:
        error_detail = "Unknown error"
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except Exception:
            error_detail = e.response.text
        return f"❌ Ingest failed: {error_detail}"
    except requests.RequestException as e:
        return f"❌ Connection error: {e}"


def chat(question: str, top_k: int) -> tuple[str, list[str]]:
    """Send a question to the RAG pipeline."""
    if not question or not question.strip():
        return "⚠️ Please enter a question.", []

    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"question": question.strip(), "top_k": int(top_k)},
            timeout=REQUEST_TIMEOUT,
        )

        if r.status_code == 400:
            # Graceful error (e.g., empty index)
            error_detail = r.json().get("detail", r.text)
            return f"⚠️ {error_detail}", []

        r.raise_for_status()
        j = r.json()

        answer = j["answer"]
        evidence_paths = [e["image_path"] for e in j["evidence"]]

        return answer, evidence_paths

    except requests.HTTPError as e:
        error_detail = "Unknown error"
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except Exception:
            error_detail = e.response.text
        return f"❌ Query failed: {error_detail}", []
    except requests.RequestException as e:
        return f"❌ Connection error: {e}", []


# --- UI Layout ---

with gr.Blocks(title="Local Page-First MMRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 📚 Local Page-First Multimodal RAG
        **PDF → Page Images → Retrieve Pages → VLM Answer + Evidence**

        *Demo limits: max {MAX_PAGES} pages, max {MAX_FILE_SIZE_MB}MB file size*
        """
    )

    # API status
    api_status = gr.Textbox(label="API Status", interactive=False)
    demo.load(fn=check_api_health, outputs=[api_status])

    # Ingest section
    gr.Markdown("## 📤 Upload Document")
    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_btn = gr.Button("Ingest", variant="primary")
    ingest_status = gr.Textbox(label="Ingest Status", interactive=False)

    ingest_btn.click(fn=ingest_pdf, inputs=[pdf], outputs=[ingest_status])

    # Chat section
    gr.Markdown("## 💬 Ask Questions")
    q = gr.Textbox(label="Question", placeholder="What does this document say about...?")
    top_k = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Top-K Evidence Pages")

    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
    gallery = gr.Gallery(label="Evidence Pages", columns=3, height=600, object_fit="contain")

    ask_btn.click(fn=chat, inputs=[q, top_k], outputs=[answer_box, gallery])

    # Allow Enter key to submit question
    q.submit(fn=chat, inputs=[q, top_k], outputs=[answer_box, gallery])


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("UI_HOST", "0.0.0.0"),
        server_port=int(os.getenv("UI_PORT", "8081")),
        show_error=True,
    )
