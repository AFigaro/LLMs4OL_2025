This is IRIS team solution for LLMs4OL 2025 Challenge. This is an initial commit which is missing requirements (to be added soon) 

## 📁 Repository Layout & Key Scripts

This repository contains the **IRIS team** solution for the **LLMs4OL 2025 Challenge**.  
All task-specific code lives under `LLMs4OL/src/`, organised by sub-task (B, C, D) and ontology
(MatOnto, OBI, SWEET).

<code>
LLMs4OL_2025/
├── LLMs4OL/
│   └── src/
│       ├── TaskB/                      # Term-typing (multi-label classification)
│       │   ├── matonto/
│       │   │   ├── augment_data_matonto.py           # surface-form variants & synonym cache
│       │   │   ├── collect_matonto_term_defs.py      # scrape / generate type definitions
│       │   │   ├── train_inference_matonto_configurable.py
│       │   │   │   ├── toggles data-augmentation / definitions
│       │   │   │   └── trains BERT/DeBERTa + threshold tuning (micro-F1)
│       │   │   └── *term_definitions.json  /  train_augmented.jsonl
│       │   ├── obi/                     # same trio for OBI
│       │   └── sweet/                   # same trio + train_deberta_sweet_cls.py
│       │
│       ├── TaskC/                      # Taxonomy discovery (parent–child edges)
│       │   ├── matonto/
│       │   │   ├── fetch_matonto_defs.py            # gather extra definitions
│       │   │   ├── filter_candidates_matonto.py     # FAISS + SBERT semantic filtering
│       │   │   └── matonto_train_and_infer_configurable.py
│       │   ├── obi/                     # analogous scripts
│       │   └── sweet/                   # analogous scripts
│       │
│       └── TaskD/
│           └── sweet/                   # Non-taxonomic relation extraction
│               ├── build_sweetD_candidates_semantic_only.py
│               ├── sweet_RE_train_and_infer_configurable.py
│               └── candidates_sweetD_{train,test}.json
│
├── LICENSE            # Apache-2.0
└── README.md
</code>

### 🔑 Script Cheat-Sheet

| Script | What it does | Output |
|--------|--------------|--------|
| **augment_data_\*** | Augments training terms with unit symbols, plural/-case variants, Wiktionary & GPT-4o ≤3-word synonyms (all cached). | `train_augmented.jsonl` |
| **collect_\*term_defs.py** | Mines compact English definitions (LLM + heuristics) for each ontology type. | `*_term_definitions.json` |
| **train_inference_\*_configurable.py** | End-to-end pipeline<br>▪ build stratified splits → tokenize → train BERT/DeBERTa multi-label classifier<br>▪ dev-set threshold sweep to maximise micro-F1<br>▪ writes submission JSON. | model checkpoints, `preds_*.json` |
| **filter_candidates_\*** | Embeds candidate pairs in SBERT + FAISS IP index, prunes to top-K semantic matches. | `candidates_*.json` |
| **\*train_and_infer_configurable.py** (Task C/D) | Fine-tunes a classification/ranking model on filtered candidates; ties into same threshold-sweep logic for predictions. | model checkpoints, submission JSON |

> **Note:** Raw LLMs4OL 2025 datasets are **not** committed. Place the original task folders under `LLMs4OL/2025/` before running any script.

---

Feel free to tell me if you want a longer, tutorial-style README (installation, GPU requirements, run commands, citation) or a diagram of the end-to-end pipeline—happy to flesh it out!
::contentReference[oaicite:0]{index=0}
