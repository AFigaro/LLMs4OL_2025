# ───────────────────────────────────────────────────────────────────────────
# SWEET · Task-C · Taxonomy Discovery
#
#   1. candidates_sweetC_train.json  – keys = train types,
#                                      values = 100 nearest *train* types
#   2. candidates_sweetC_test.json   – keys = test  types,
#                                      values = 100 nearest *test*  types
#
# No type ever crosses the split.
# ───────────────────────────────────────────────────────────────────────────

import json, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── paths -----------------------------------------------------------------
ROOT        = Path("LLMs4OL")

TRAIN_TXT   = ROOT / "2025/TaskC-TaxonomyDiscovery/SWEET/train/sweet_train_types.txt"
TEST_TXT    = ROOT / "2025/TaskC-TaxonomyDiscovery/SWEET/test/sweet_test_types.txt"

OUT_DIR     = ROOT / "src/TaskC/sweet"
OUT_TRAIN   = OUT_DIR / "candidates_sweetC_train.json"
OUT_TEST    = OUT_DIR / "candidates_sweetC_test.json"

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 100

# ── helper ----------------------------------------------------------------
def make_cand_map(type_list, top_k, model):
    """Return {type: [top-k neighbours chosen *within* type_list]}."""
    types = sorted(type_list)
    embs  = model.encode(types, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs.astype(np.float32))

    cand = {}
    for i, vec in tqdm(enumerate(embs), total=len(embs), desc="FAISS search"):
        _, idx = index.search(vec[None, :].astype(np.float32), top_k + 1)
        cand[types[i]] = [types[j] for j in idx[0] if types[j] != types[i]][:top_k]
    return cand

# ── load split lists ------------------------------------------------------
train_types = [t.strip() for t in TRAIN_TXT.read_text("utf-8").splitlines() if t.strip()]
test_types  = [t.strip() for t in TEST_TXT.read_text("utf-8").splitlines() if t.strip()]

print(f"[INFO] train types : {len(train_types):,}")
print(f"[INFO] test  types : {len(test_types):,}")

# ── embed & build separately ---------------------------------------------
model = SentenceTransformer(MODEL_NAME)

print("[INFO] building train-to-train neighbours …")
cand_train = make_cand_map(train_types, TOP_K, model)

print("[INFO] building test-to-test neighbours …")
cand_test  = make_cand_map(test_types , TOP_K, model)

# ── save ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRAIN.write_text(json.dumps(cand_train, indent=2, ensure_ascii=False), "utf-8")
OUT_TEST .write_text(json.dumps(cand_test , indent=2, ensure_ascii=False), "utf-8")

print(f"[DONE] wrote:\n  • {OUT_TRAIN}\n  • {OUT_TEST}")
