# ───────────────────────────────────────────────────────────────────────────
# SWEET · Task-D · Non-Taxonomic RE
#
# Produces two JSON files:
#   1. candidates_sweetD_train.json   – keys = train types,
#                                       values = 100 nearest train types
#   2. candidates_sweetD_test.json    – keys = test  types,
#                                       values = 100 nearest test  types
#
# The two neighbour lists are computed in **completely separate spaces**:
#   − Train neighbours are found only within the train set
#   − Test  neighbours are found only within the test  set
# ───────────────────────────────────────────────────────────────────────────

import json, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────
ROOT        = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_TXT   = ROOT / "2025/TaskD-NonTaxonomicRE/SWEET/train/sweet_train_re_types.txt"
TEST_TXT    = ROOT / "2025/TaskD-NonTaxonomicRE/SWEET/test/sweet_test_re_types.txt"

OUT_DIR     = ROOT / "src/TaskD/sweet"
OUT_TRAIN   = OUT_DIR / "candidates_sweetD_train.json"
OUT_TEST    = OUT_DIR / "candidates_sweetD_test.json"

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 100

# ── helper ---------------------------------------------------------------
def build_candidates(type_list, top_k, model):
    """Return dict: {type: [top-k neighbours from *type_list*]}"""
    types = sorted(type_list)
    embs  = model.encode(types, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))

    cand_map = {}
    for i, vec in tqdm(enumerate(embs), total=len(embs), desc="FAISS search"):
        _, idx = index.search(vec[None, :].astype(np.float32), top_k + 1)
        cand_map[types[i]] = [types[j] for j in idx[0] if types[j] != types[i]][:top_k]
    return cand_map

# ── load lists -----------------------------------------------------------
train_types = [t.strip() for t in TRAIN_TXT.read_text("utf-8").splitlines() if t.strip()]
test_types  = [t.strip() for t in TEST_TXT.read_text("utf-8").splitlines() if t.strip()]
print(f"[INFO] train types: {len(train_types):,}")
print(f"[INFO] test  types: {len(test_types):,}")

# ── embed & build neighbours separately ----------------------------------
model = SentenceTransformer(MODEL_NAME)

print("[INFO] building train-to-train neighbours …")
cand_train = build_candidates(train_types, TOP_K, model)

print("[INFO] building test-to-test neighbours …")
cand_test  = build_candidates(test_types , TOP_K, model)

# ── save -----------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRAIN.write_text(json.dumps(cand_train, indent=2, ensure_ascii=False), "utf-8")
OUT_TEST .write_text(json.dumps(cand_test , indent=2, ensure_ascii=False), "utf-8")
print(f"[DONE] wrote:\n  • {OUT_TRAIN}\n  • {OUT_TEST}")
