# ───────────────────────────────────────────────────────────────────────────
# MatOnto · Task-C
#
# Produces two Top-100 neighbour maps:
#   1. candidates_matonto_train.json   (train→train)
#   2. candidates_matonto_test.json    (test →test)
#
# Each neighbour list is computed strictly within its own split.
# ───────────────────────────────────────────────────────────────────────────

import json, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ───────── CONFIG ─────────────────────────────────────────────────────────
ROOT        = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_TXT   = ROOT / "2025/TaskC-TaxonomyDiscovery/MatOnto/train/matonto_train_types.txt"
TEST_TXT    = ROOT / "2025/TaskC-TaxonomyDiscovery/MatOnto/test/matonto_test_types.txt"

OUT_DIR     = ROOT / "src/TaskC/matonto"
OUT_TRAIN   = OUT_DIR / "candidates_matonto_train.json"
OUT_TEST    = OUT_DIR / "candidates_matonto_test.json"

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 100

# ───────── HELPERS ────────────────────────────────────────────────────────
def build_map(type_list, top_k, model):
    """Return {t: [top-k neighbours drawn only from `type_list`]}."""
    types = sorted(type_list)
    embs  = model.encode(types, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs.astype(np.float32))

    cand = {}
    for i, vec in tqdm(enumerate(embs), total=len(embs), desc="Neighbours"):
        _, idx = index.search(vec[None, :].astype(np.float32), top_k + 1)
        cand[types[i]] = [types[j] for j in idx[0] if types[j] != types[i]][:top_k]
    return cand

# ───────── LOAD SPLITS ────────────────────────────────────────────────────
train_types = [t.strip() for t in TRAIN_TXT.read_text("utf-8").splitlines() if t.strip()]
test_types  = [t.strip() for t in TEST_TXT.read_text("utf-8").splitlines() if t.strip()]

print(f"[INFO] train types : {len(train_types):,}")
print(f"[INFO] test  types : {len(test_types):,}")

# ───────── EMBED & BUILD SEPARATELY ───────────────────────────────────────
print(f"[INFO] embedding with '{MODEL_NAME}' …")
model = SentenceTransformer(MODEL_NAME)

print("[INFO] building train→train neighbours …")
cand_train = build_map(train_types, TOP_K, model)

print("[INFO] building test→test  neighbours …")
cand_test  = build_map(test_types , TOP_K, model)

# ensure non-empty lists (very rare)
for t, lst in cand_train.items():
    if not lst:
        cand_train[t] = [n for n in train_types if n != t][:TOP_K]
for t, lst in cand_test.items():
    if not lst:
        cand_test[t] = [n for n in test_types if n != t][:TOP_K]

# ───────── SAVE ───────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRAIN.write_text(json.dumps(cand_train, indent=2, ensure_ascii=False), "utf-8")
OUT_TEST .write_text(json.dumps(cand_test , indent=2, ensure_ascii=False), "utf-8")

print(f"[DONE] wrote:\n  • {OUT_TRAIN}\n  • {OUT_TEST}")
