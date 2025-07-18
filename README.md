## 📁 Repository Layout & Key Scripts

This repository contains the **IRIS team** solution for the **LLMs4OL 2025 Challenge**.  
All task-specific code lives under `LLMs4OL/src/`, organised by sub-task (B, C, D) and ontology
(MatOnto, OBI, SWEET).


<code> LLMs4OL_2025/
├── LLMs4OL/
│   └── src/
│       ├── TaskB/                      # Term-typing (multi-label classification)
│       │   ├── matonto/
│       │   │   ├── augment_data_matonto.py           # surface-form variants & synonym cache
│       │   │   ├── collect_matonto_term_defs.py      # scrape / generate type definitions
│       │   │   ├── train_inference_matonto_configurable.py   # trains and makes predictions
│       │   │   └── *term_definitions.json  /  train_augmented.jsonl
│       │   ├── obi/                     # same trio for OBI
│       │   └── sweet/                   # same trio + train_deberta_sweet_cls.py
│       │
│       ├── TaskC/                      # Taxonomy discovery (parent–child edges)
│       │   ├── matonto/
│       │   │   ├── fetch_matonto_defs.py            # gather extra definitions
│       │   │   ├── filter_candidates_matonto.py     # FAISS + SBERT semantic filtering
│       │   │   └── matonto_train_and_infer_configurable.py    # trains and makes predictions
│       │   ├── obi/                     # analogous scripts
│       │   └── sweet/                   # analogous scripts
│       │
│       └── TaskD/
│           └── sweet/                   # Non-taxonomic relation extraction
│               ├── build_sweetD_candidates_semantic_only.py
│               ├── sweet_RE_train_and_infer_configurable.py  #trains and makes predictions
│               └── candidates_sweetD_{train,test}.json
│
├── LICENSE            # Apache-2.0
└── README.md
</code>

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/AFigaro/LLMs4OL_2025.git
cd LLMs4OL_2025

# 2. Create + activate a Python 3.10.12 conda env
conda create -n llms4ol python=3.10.12 -y
conda activate llms4ol

# 3. Install project dependencies
pip install -r requirements.txt
```

## 🔑 Script Cheat-Sheet

| Script | Subtask(s) | What it does | Output |
|--------|------------|--------------|--------|
| **augment_data_\*** | B | Augments training terms with unit symbols, plural/case variants, plus Wiktionary & GPT-4o-generated synonyms. | `train_augmented.jsonl` |
| **collect_\*term/type_defs.py** | B / C / D | Mines short English definitions for each ontology term/type. | `*_term_definitions.json` |
| **train_inference_\*_configurable.py** | B | End-to-end pipeline that trains a DeBERTa multi-label classifier and writes predictions to a submission file. | model checkpoints, `preds_*.json` |
| **filter_candidates_\*** | C & D | Embeds candidate pairs with SBERT + FAISS, then prunes to the top-K semantic matches. | `candidates_*.json` |
| **\*train_and_infer_configurable.py** | C & D | Fine-tunes a classification/ranking model on filtered candidates; includes threshold-sweep logic for predictions. | model checkpoints, submission JSON |


> **Note:** Raw LLMs4OL 2025 datasets are **not** committed. Place the original task folders under `LLMs4OL/2025/` before running any script.

