# TravelMind 🌍

**A domain-specific travel chatbot comparing Retrieval-Augmented Generation (RAG) against local LLM fine-tuning.**

Built as an NLP course project, TravelMind implements two complementary approaches to the same problem — answering travel Q&A grounded in a fixed knowledge base — and empirically compares them across six evaluation metrics.

## Why this project

Generic LLMs hallucinate on niche or specific facts (prices, visa rules, seasonal advice). This project asks: what's the better fix — retrieving relevant context at inference time (RAG), or baking domain knowledge into the model's weights (fine-tuning)? Both were built on the same dataset and evaluated head-to-head.

## Architecture

**RAG Pipeline (Groq)**
```
User Query → Embed (all-mpnet-base-v2) → FAISS vector search (top-3) →
Structured prompt → Llama 3.1-8B (Groq API) → Grounded answer
```

**Fine-Tuned Local Model (TinyLlama)**
```
User Query → Tokenize → TinyLlama 1.1B + LoRA adapters →
Decode → Answer (no API needed)
```

## Tech Stack

`Python` `sentence-transformers` `FAISS` `Groq API (Llama 3.1-8B)` `Unsloth` `TinyLlama 1.1B` `LoRA` `TRL / SFTTrainer` `Gradio` `BLEU / ROUGE`

## Results

Both systems evaluated on a 10-query test set:

| Metric | Groq RAG | TinyLlama FT | Winner |
|---|---|---|---|
| BLEU-1 | 0.1308 | 0.0775 | Groq RAG (+69%) |
| BLEU-2 | 0.0763 | 0.0255 | Groq RAG (+199%) |
| BLEU-4 | 0.0396 | 0.0079 | Groq RAG (+401%) |
| ROUGE-1 | 0.2008 | 0.1384 | Groq RAG (+45%) |
| ROUGE-2 | 0.0758 | 0.0221 | Groq RAG (+243%) |
| ROUGE-L | 0.1606 | 0.1082 | Groq RAG (+48%) |

**Key finding:** RAG outperformed fine-tuning on every metric. This isn't surprising given the constraints (only 1,000 training samples, 100 training steps on a free-tier T4 GPU) — the fine-tuning pipeline is architecturally correct, but under-trained. Full analysis, including a mechanistic breakdown of *why* the training loss oscillated instead of decreasing, is in the [full report](docs/TravelMind_Report_v2.pdf.pdf).

## Dataset

[`JasleenSingh91/travel-QA`](https://huggingface.co/datasets/JasleenSingh91/travel-QA) — ~17,000 Q&A pairs on destinations, visas, weather, and travel costs. Two separate preprocessing pipelines were used (1,500 samples for RAG indexing, 1,000 instruction-formatted samples for fine-tuning).

## Key implementation details

- **Embeddings:** `all-mpnet-base-v2`, 768-dim vectors, brute-force FAISS `IndexFlatL2` search
- **Hallucination guardrail:** prompt explicitly instructs the model to say "not in dataset" rather than fabricate — verified working (see report, Figure 2)
- **LoRA setup:** rank 16, applied to 7 projection layers, only 4.5M / ~1.1B parameters (0.41%) trained
- **Quantization:** 4-bit loading via Unsloth, ~4x memory reduction, enabling training on a free Colab T4

## Future Improvements

- Train TinyLlama for 300+ steps / full epochs for a meaningful loss reduction
- Add stricter prompt constraints to prevent RAG from leaking pretrained knowledge on out-of-domain queries
- Evaluate on a larger test set (50+ queries) for statistically reliable scores
- Persist the FAISS index to disk instead of rebuilding every session
- Explore using the fine-tuned TinyLlama as the generator inside the RAG pipeline

## Run it yourself

```bash
pip install -q datasets sentence-transformers faiss-cpu groq gradio rouge-score transformers unsloth trl accelerate torch
export GROQ_API_KEY="your-key-here"
```

Then open `TravelMind.ipynb` and run cells top to bottom. The RAG section launches a Gradio interface for interactive querying; the fine-tuning section (Colab + GPU recommended) trains and serves the TinyLlama model.

## Full Report

See [`docs/TravelMind_Report_v2.pdf.pdf`](docs/TravelMind_Report_v2.pdf.pdf) for the complete write-up, including architecture diagrams, the training loss curve, and detailed metric interpretation.

---
*Author: Ashwath Bhuyan — DTU, Information Technology*
