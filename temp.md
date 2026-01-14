If the baseline demo (PDF → page PNGs → retrieval → UI evidence → answer stub/LLM) is working end-to-end, the next steps are to (1) harden the pipeline so it is repeatable and demo-safe, and then (2) upgrade retrieval and generation to the agreed “best multimodal experience” stack while keeping the same API/UI contracts.

## Priority 1 — Make the demo robust (no surprises)

1. **Idempotent ingest (no duplicated vectors)**

   * Add `docs/<doc_id>/manifest.json` with `{sha256, num_pages, indexed_at, index_backend, embedder_id}`
   * On ingest: if `indexed=true`, skip re-adding to FAISS.

2. **Persistence contract**

   * Ensure index metadata and FAISS index can be deleted/rebuilt deterministically:

     * `scripts/reset_index.sh` (clears `data/mmrag/index/`)
     * `scripts/reindex_all.sh` (walk docs, rebuild index)

3. **Observability**

   * Add `/stats` endpoint: number of docs, pages indexed, embedding dim, index type, device, GPU name, memory usage (best-effort).
   * Add structured logs: ingest time per stage (pdf render / embed / add-to-index).

4. **Constraints & safety**

   * Enforce demo limits: 40 pages, max DPI, max file size, allowed PDF only.
   * Graceful error messages returned to UI.

## Priority 2 — Upgrade generation to Qwen2.5-VL on GPU

5. **Replace stub with Transformers-based Qwen2.5-VL-7B-Instruct**

   * Keep interface: `answer(question, evidence_images) -> str`
   * Implement “evidence-conditioned” prompting (pass the top-K page images to the VLM).
   * Add a hard fallback to stub if model load fails.

6. **Add “grounding discipline”**

   * Include page citations in the answer: “(evidence: p. 3, p. 7)”
   * Optional: return a short “why these pages” rationale.

## Priority 3 — Upgrade retrieval toward ColPali / ColQwen2 style

7. **Swap embedder implementation**

   * Keep the same embedder API, change internals from CLIP to ColPali/ColQwen2.
   * Reindex once; do not touch FastAPI or UI.

8. **Improve ranking**

   * If ColPali supports late-interaction scoring, store per-page representations accordingly.
   * Keep FAISS as a coarse stage if needed; re-rank with ColPali scoring.

## Priority 4 — vLLM serving (only if it installs cleanly)

9. **Add a vLLM “probe” and optional serving mode**

   * If vLLM installs and runs on aarch64+GB10:

     * Run vLLM on `localhost:8000`
     * FastAPI `/chat` calls vLLM
   * If not, keep Transformers (the demo remains fully valid).

---

### What I recommend you do next (minimal, high value)

1. Implement **idempotent ingest + manifests**
2. Add **/stats** + timing logs
3. Wire **Qwen2.5-VL via Transformers** (GPU) with fallback
4. Then replace embeddings with **ColPali/ColQwen2** and reindex

If you paste your current repo tree (or just confirm whether you already wrote manifests / reset scripts), I will give you exact patch-level code for steps 1–4 without changing ports or architecture.
