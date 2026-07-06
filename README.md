# Hallucination Detection Engine for Financial LLM Outputs

**Thomson Reuters · Trustworthy AI Research**

An end-to-end system for detecting hallucinations in LLM-generated financial content by verifying claims against SEC filing evidence using a fine-tuned NLI model.

---

## Overview

Financial LLM outputs — summaries, analyst notes, generated commentary — often contain claims that are subtly wrong: inflated growth figures, misattributed statements, contradicted risk disclosures. This pipeline builds a retrieval-augmented verification system that:

1. Ingests SEC filings (10-K, 10-Q, 8-K) from EDGAR or local PDFs
2. Parses and chunks them into section-aware, retrievable units
3. Embeds and indexes chunks in a vector database
4. Retrieves supporting evidence for any claim
5. Uses human annotation to build a labeled NLI dataset
6. Fine-tunes a DeBERTa model to classify claim/evidence pairs as **Entailment**, **Neutral**, or **Contradiction**
7. Flags contradicted or unverifiable claims as likely hallucinations

```
SEC Filings → Parse → Chunk → Embed → Index → Retrieve → Annotate
                                                              │
                                                              ▼
                                          NLI Dataset → Fine-tune DeBERTa
                                                              │
                                                              ▼
                                    Evaluate → Inference Pipeline → Demo
```

---

## Notebook

**`hallucination_detection_complete.ipynb`** — single notebook, 14 sections, runs sequentially top to bottom.

| # | Section | Output |
|---|---------|--------|
| 1 | Environment Setup | Pinned dependencies, logger, seeded RNG, device detection |
| 2 | Configuration | `PipelineConfig` dataclass + project directory tree |
| 3 | SEC Ingestion | `data/raw_document_manifest.parquet` |
| 4 | PDF Parsing | `data/parsed/<ticker>_<form>_<date>.json` |
| 5 | Section-aware Chunking | `data/chunks/chunks.parquet` |
| 6 | Embedding Generation | `data/embeddings/embeddings.npy` + `metadata.parquet` |
| 7 | Vector Database | `data/vector_db/faiss.index` or `chroma/` |
| 8 | Retrieval Pipeline | `RetrievalPipeline.retrieve(claim)` |
| 9 | Annotation Interface | Gradio UI + `data/annotations/annotations.db` |
| 10 | Dataset Builder | `train.csv` / `validation.csv` / `test.csv` |
| 11 | DeBERTa Fine-tuning | `models/fine_tuned_model/` |
| 12 | Evaluation | `outputs/evaluation_report.json`, confusion matrix, ROC curves |
| 13 | Inference Pipeline | `HallucinationDetector.detect(summary)` |
| 14 | End-to-End Demo | `outputs/e2e_demo_report.{json,html}` |

Every section persists concrete artefacts to disk, so any section can be re-run independently once the pipeline has completed once. Each section exposes a `SKIP_<SECTION>` boolean flag to reload prior outputs instead of recomputing.

---

## Project Structure

```
project/
├── data/
│   ├── raw_documents/       # Downloaded / manually placed SEC filings
│   ├── parsed/               # Structured ParsedDocument JSON per filing
│   ├── chunks/                # chunks.parquet
│   ├── embeddings/           # embeddings.npy + metadata.parquet
│   ├── vector_db/             # faiss.index or chroma/
│   └── annotations/          # annotations.db, train/validation/test.csv
├── models/
│   └── fine_tuned_model/     # Fine-tuned DeBERTa weights, tokenizer, label_map.json
├── outputs/                    # Evaluation reports, plots, demo reports
├── logs/                        # Timestamped run logs
└── hallucination_detection_complete.ipynb
```

---

## Key Models

| Component | Model | Why |
|-----------|-------|-----|
| Embeddings | `sentence-transformers/all-mpnet-base-v2` | Best MTEB score among ≤125M-param general-purpose encoders; 768-dim, Apache-2.0 |
| Reranker (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight precision boost over bi-encoder retrieval |
| NLI Classifier | `cross-encoder/nli-deberta-v3-small` | Pretrained NLI head — fine-tuning refines rather than trains from scratch |

> **Note:** `microsoft/deberta-v3-large` is a masked-LM backbone with **no NLI head**. Using it directly requires a large labeled corpus (thousands of examples) to train a classification head from random initialization — the small annotation set produced by Section 9's demo workflow is insufficient for this. `cross-encoder/nli-deberta-v3-small` ships with a pretrained 3-class NLI head, making it the correct choice for fine-tuning on a small, growing annotation set.

---

## Label Convention

```python
LABEL2ID = {'CONTRADICTION': 0, 'ENTAILMENT': 1, 'NEUTRAL': 2}
```

This ordering matches the base model's native `id2label` mapping. **Do not reorder this dict** unless the underlying `NLI_MODEL` is also changed — mismatched label ordering silently collapses all predictions into a single class (see `docs/known-issues.md` / troubleshooting below).

All label decoding in inference reads from `model.config.id2label`, not the global `LABEL2ID` dict, so the system stays correct even if a different checkpoint is swapped in.

---

## Quick Start

1. Open `hallucination_detection_complete.ipynb`.
2. Run all cells top to bottom. On first run:
   - Section 3 downloads sample SEC filings for `AAPL` and `MSFT` (edit `DEMO_TICKERS`/`DEMO_FORMS` to change).
   - Section 9 seeds a small synthetic annotation set so Sections 10–14 have data to work with even before real annotation happens.
   - Section 11 fine-tunes the NLI model — expect this to take longer on CPU.
3. To verify a custom LLM summary, edit `DEMO_LLM_SUMMARY` in Section 14 and re-run cells 14 onward.
4. To annotate manually, set `LAUNCH_GRADIO = True` in Section 9 to open the annotation UI.

### Re-running a single section

Each data-producing section has a `SKIP_<NAME>` flag near its execution cell (e.g. `SKIP_CHUNKING`, `SKIP_EMBEDDING`, `SKIP_INDEXING`, `SKIP_TRAINING`). Set to `True` to reload cached artefacts from disk instead of recomputing.

---

## Using the Inference Pipeline

```python
detector = HallucinationDetector(
    model=nli_model,
    tokenizer=nli_tokenizer,
    retrieval=retrieval_pipeline,
    config=cfg,
    contradiction_only=False,     # also flag NEUTRAL (unverifiable) claims
    confidence_threshold=0.50,
)

verdicts = detector.detect(
    summary="Apple's iPhone revenue grew 15% to $250 billion in fiscal 2023...",
    ticker_filter="AAPL",         # optional: restrict evidence search
)

for v in verdicts:
    print(v.claim, "→", v.prediction, f"(conf={v.confidence:.2f})",
          "🚨 FLAGGED" if v.hallucination_flag else "OK")
```

Or run the full orchestrated pipeline:

```python
e2e = EndToEndPipeline(config=cfg, detector=detector)
result = e2e.run(summary=my_llm_output, ticker="AAPL", top_k=3)
print(result['stats'])
```

---

## Production Hardening Checklist

- [ ] Replace the synthetic seed corpus with ≥1,000 human-annotated (claim, evidence, label) pairs
- [ ] Enable PostgreSQL for `cfg.ANNOTATION_DB` in multi-annotator deployments
- [ ] Enable cross-encoder reranking (`use_reranker=True`) for higher-precision retrieval
- [ ] Add inter-annotator agreement metrics (Cohen's κ) in Section 10
- [ ] Integrate Weights & Biases or MLflow for training experiment tracking
- [ ] Package `HallucinationDetector` as a FastAPI endpoint for upstream consumption
- [ ] Schedule periodic re-training as new SEC filings arrive
- [ ] Evaluate `deberta-v3-base` vs. `deberta-v3-small` NLI variants for cost/accuracy trade-off at scale

---

## Troubleshooting

**All predictions come back as the same class with near-constant, low confidence.**
This is almost always a label-mapping mismatch between `LABEL2ID`/`ID2LABEL` and the loaded model's `config.id2label`. Confirm:
- `NLI_MODEL` in `PipelineConfig` points to a model with a pretrained NLI head (not a bare MLM backbone).
- `ignore_mismatched_sizes` is `False` when loading any checkpoint that already has a 3-class head — setting it `True` silently reinitializes the classification layer.
- Inference code decodes via `model.config.id2label[pred_id]`, not a hardcoded global dict.

**Retrieval returns no results.**
Check that Section 6/7 completed and `cfg.VECTOR_DB_PATH` contains either `faiss.index` or a `chroma/` directory matching `cfg.VECTOR_BACKEND`.

**Gradio annotation UI doesn't open.**
Set `LAUNCH_GRADIO = True` in Section 9 and re-run that cell; it launches on `localhost:7860` by default.

---

## Dependencies

```
transformers, sentence-transformers, datasets, accelerate, evaluate, torch,
faiss-cpu, chromadb, pdfplumber, PyMuPDF, pandas, numpy, scikit-learn,
gradio, sqlalchemy, psycopg2-binary, tqdm, huggingface_hub,
sec-edgar-downloader, pyarrow
```

See Section 1 of the notebook for pinned versions.

---

## License & Usage

Internal Thomson Reuters research artefact. All models used are open-source (Apache-2.0 / MIT) via Hugging Face. SEC filing data is public domain per EDGAR's terms of use.
