# ───────────────────────────────────────────────────────────────────────────
# Four variants controlled by:
#   USE_DEFS  – True / False            (add definitions?)
#   NEG_SRC   – "candidates" | "random" (negative parent source)
# Uses split-specific neighbour files:
#   • candidates_sweetC_train.json   – neighbours only among train types
#   • candidates_sweetC_test.json    – neighbours only among test  types
# ───────────────────────────────────────────────────────────────────────────

# ╭── variant switches ─────────────────────────────────────────────────────╮
USE_DEFS  = True                # include definitions in prompts?
NEG_SRC   = "candidates"        # "candidates" | "random"
NEG_PER_CHILD = 9               # always 9 negatives / child
# ╰──────────────────────────────────────────────────────────────────────────╯

import os, json, random, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch, torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.special import softmax

# ───────── PATHS ──────────────────────────────────────────────────────────
ROOT = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
PAIRS_JSON   = ROOT / "2025/TaskC-TaxonomyDiscovery/SWEET/train/sweet_train_pairs.json"
TRAIN_TYPES  = ROOT / "2025/TaskC-TaxonomyDiscovery/SWEET/train/sweet_train_types.txt"
TEST_TYPES   = ROOT / "2025/TaskC-TaxonomyDiscovery/SWEET/test/sweet_test_types.txt"

DEFS_JSON    = ROOT / "src/TaskC/sweet/sweet_types_leads.json"
CAND_TRAIN   = ROOT / "src/TaskC/sweet/candidates_sweetC_train.json"
CAND_TEST    = ROOT / "src/TaskC/sweet/candidates_sweetC_test.json"

TAG      = f"sweet_{'def' if USE_DEFS else 'str'}_{NEG_SRC}"
OUT_DIR  = ROOT / f"src/TaskC/sweet/ckpt_deberta_v3_{TAG}"
PRED_JS  = ROOT / f"src/TaskC/sweet/pred_edges_{TAG}.json"
MODEL_ID = "microsoft/deberta-v3-large"

# ───────── HYPERPARAMS ────────────────────────────────────────────────────
SEED, BATCH, EPOCHS, LR = 42, 4, 4, 1e-5
MAX_LEN   = 512 if USE_DEFS else 128
DEV_FR, TEST_FR, INF_BATCH = 0.10, 0.10, 4

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── LOAD CORPUS ────────────────────────────────────────────────────
pairs        = json.loads(PAIRS_JSON.read_text("utf-8"))
train_types  = [t.strip() for t in TRAIN_TYPES.read_text("utf-8").splitlines() if t.strip()]
test_types   = [t.strip() for t in TEST_TYPES.read_text("utf-8").splitlines() if t.strip()]

defs_map     = json.loads(DEFS_JSON.read_text("utf-8")) if USE_DEFS else {}
cand_train   = json.loads(CAND_TRAIN.read_text("utf-8")) if NEG_SRC=="candidates" else {}
cand_test    = json.loads(CAND_TEST.read_text("utf-8"))  if NEG_SRC=="candidates" else {}

gold = {}
for e in pairs:
    gold.setdefault(e["child"], set()).add(e["parent"])

children = list(gold); random.shuffle(children)
n_tot = len(children); n_dev = math.floor(DEV_FR*n_tot); n_tst = math.floor(TEST_FR*n_tot)
dev_children   = set(children[:n_dev])
test_ch_train  = set(children[n_dev:n_dev+n_tst])   # held-out for metrics
train_children = set(children[n_dev+n_tst:])

print(f"[INFO] child split train/dev/test = {len(train_children)}/{len(dev_children)}/{len(test_ch_train)}")

# ───────── PROMPT MAKER ───────────────────────────────────────────────────
def mk_prompt(c, p):
    if USE_DEFS:
        return (f"[CHILD] {c}\n[DEF] {defs_map.get(c,'')}\n\n"
                f"[PARENT] {p}\n[DEF] {defs_map.get(p,'')}")
    return f"[CHILD] {c}\n\n[PARENT] {p}"

# ───────── BUILD DATASETS ─────────────────────────────────────────────────
def build(child_set, tag):
    data = []
    for c in child_set:
        pos_parents = gold[c]
        if NEG_SRC == "candidates":
            pool = [t for t in cand_train.get(c, []) if t not in pos_parents and t != c]
        else:
            pool = [t for t in train_types if t not in pos_parents and t != c]

        negs = random.sample(pool, min(NEG_PER_CHILD, len(pool)))
        data += [{"text": mk_prompt(c, p), "label": 1} for p in pos_parents]
        data += [{"text": mk_prompt(c, p), "label": 0} for p in negs]
    print(f"[{tag}] examples = {len(data):,}")
    return data

train_data = build(train_children, "train")
dev_data   = build(dev_children,   "dev")
test_data  = build(test_ch_train,  "test")

# ───────── TOKENISATION ───────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_ID)
def enc(b): return tok(b["text"], truncation=True, max_length=MAX_LEN)
train_ds = Dataset.from_list(train_data).map(enc, batched=True, remove_columns=["text"])
dev_ds   = Dataset.from_list(dev_data  ).map(enc, batched=True, remove_columns=["text"])
test_ds  = Dataset.from_list(test_data ).map(enc, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tok)

# ───────── MODEL & TRAINER ────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(device)
pos = sum(e["label"] for e in train_data); neg = len(train_data) - pos
pos_w = neg / pos

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        outs   = model(**inputs)
        loss   = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=outs.logits.device))(outs.logits, labels)
        if return_outputs:
            detached = {k:(v.detach() if isinstance(v, torch.Tensor) else v) for k, v in outs.items()}
            return loss, outs.__class__(**detached)
        return loss

def metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f}

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=32,
    num_train_epochs=EPOCHS,
    fp16=True,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    save_total_limit=2,
)

trainer = WTrainer(model=model, args=args,
                   train_dataset=train_ds, eval_dataset=dev_ds,
                   tokenizer=tok, data_collator=collator,
                   compute_metrics=metrics)
trainer.train(resume_from_checkpoint=True)

model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)

# ───────── τ tuning & save ────────────────────────────────────────────────
dev_probs = softmax(trainer.predict(dev_ds).predictions, axis=-1)[:, 1]
tau, _ = max(
    ((thr, precision_recall_fscore_support(dev_ds["label"], (dev_probs >= thr).astype(int),
                                           average="binary", zero_division=0)[2])
     for thr in np.arange(0.05, 0.951, 0.01)),
    key=lambda x: x[1]
)
(OUT_DIR / "best_threshold.txt").write_text(f"{tau:.4f}\n", "utf-8")
print(f"[DEV] τ = {tau:.2f}")

# ───────── internal held-out test metrics ─────────────────────────────────
probs_test = softmax(trainer.predict(test_ds).predictions, axis=-1)[:, 1]
test_pred  = (probs_test >= tau).astype(int)
p, r, f, _ = precision_recall_fscore_support(test_ds["label"], test_pred, average="binary", zero_division=0)
print(f"[TEST internal] F1={f:.3f}  P={p:.3f}  R={r:.3f}")

# ───────── OFFICIAL TEST INFERENCE (test types only) ──────────────────────
print("[INFO] inference on official SWEET test children …")
pred_edges, texts, pairs = [], [], []

def flush():
    if not texts:
        return
    with torch.no_grad():
        probs = softmax(model(**tok(texts, padding=True, truncation=True,
                                    max_length=MAX_LEN, return_tensors="pt"
                                    ).to(device)
                              ).logits.detach().cpu().numpy(), axis=-1)[:, 1]
    for (c, p), s in zip(pairs, probs):
        if s >= tau:
            pred_edges.append({"parent": p, "child": c, "score": float(s)})
    texts.clear(); pairs.clear()

for c in tqdm(test_types, unit="child"):
    if NEG_SRC == "candidates":
        parents = cand_test.get(c, [])
    else:  # random sample among test types
        pool = [t for t in test_types if t != c]
        parents = random.sample(pool, min(NEG_PER_CHILD, len(pool)))
    for p in parents:
        texts.append(mk_prompt(c, p)); pairs.append((c, p))
        if len(texts) >= INF_BATCH:
            flush()
flush()

PRED_JS.parent.mkdir(parents=True, exist_ok=True)
PRED_JS.write_text(json.dumps(pred_edges, indent=2, ensure_ascii=False), "utf-8")
print(f"[DONE] {len(pred_edges):,} edges saved → {PRED_JS}")
