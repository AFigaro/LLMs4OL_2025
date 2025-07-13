#!/usr/bin/env python3
# run_matonto_multilabel.py
# ────────────────────────────────────────────────────────────────────────
USE_AUGMENTED   = False    # True → use train_augmented*.jsonl
USE_DEFINITIONS = False    # True → add definition sentence
# ────────────────────────────────────────────────────────────────────────

import os, json, random, collections, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers.trainer_utils import EvalPrediction

# ───────── paths ────────────────────────────────────────────────────────
ROOT = Path("LLMs4OL")

ORIG_TRAIN = ROOT / "2025/TaskB-TermTyping/MatOnto/train/term_typing_train_data.json"
ORIG_TEST  = ROOT / "2025/TaskB-TermTyping/MatOnto/test/matonto_term_typing_test_data.json"

AUG_TRAIN          = ROOT / "src/TaskB/matonto_binary/train_augmented.jsonl"
AUG_TRAIN_DEFS     = ROOT / "src/TaskB/matonto_binary/train_augmented_with_defs.jsonl"

DEF_FILE           = ROOT / "src/TaskB/matonto_binary/matonto_term_definitions.json"

TAG      = f"{'aug' if USE_AUGMENTED else 'orig'}_{'defs' if USE_DEFINITIONS else 'nodef'}"
CKPT_DIR = ROOT / f"src/TaskB/matonto_binary/checkpoints_bert/{TAG}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SUB_FILE = ROOT / f"src/TaskB/matonto_binary/preds_{TAG}.json"

MODEL_ID  = "bert-base-uncased"  # or "roberta-base", "distilbert-base-uncased", etc.
SEED      = 42
DEV_FRAC  = 0.05
BS_TRAIN  = 8
BS_EVAL   = 8
EPOCHS    = 30
LR        = 2e-5
MAX_LEN   = 512
THR_GRID  = np.linspace(0.05, 0.95, 19)
TOP_K     = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ═════ 1. build triplets ════════════════════════════════════════════════
if USE_AUGMENTED:
    aug_path = AUG_TRAIN_DEFS if USE_DEFINITIONS else AUG_TRAIN
    rows = [json.loads(l) for l in aug_path.read_text("utf-8").splitlines()]
    triplets = [
        (r["text"], r.get("definition", ""), r["label"])
        if "definition" in r else (r["text"], "", r["label"])
        for r in rows
    ]
else:
    orig = json.loads(ORIG_TRAIN.read_text("utf-8"))
    defs = json.loads(DEF_FILE.read_text("utf-8"))
    triplets = [
        (r["term"],
         defs.get(r["term"], "") if USE_DEFINITIONS else "",
         t)
        for r in orig for t in r["types"]
    ]

types_vocab = sorted({t for _, _, t in triplets})
lab2id = {t: i for i, t in enumerate(types_vocab)}
L = len(types_vocab)

def make_row(term, definition, typ):
    txt = term if not USE_DEFINITIONS else f"[TERM] {term}\n[DEF] {definition}"
    vec = [0]*L; vec[lab2id[typ]] = 1
    return {"text": txt, "labels": vec}

examples = [make_row(*tri) for tri in triplets]
random.shuffle(examples)

# ═════ 2. stratified splits ═════════════════════════════════════════════
multi = torch.tensor([e["labels"] for e in examples])
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=DEV_FRAC,
                                        random_state=SEED)
train_idx, dev_idx = next(msss.split(examples, multi))
train_pool = [examples[i] for i in train_idx]
dev_ex     = [examples[i] for i in dev_idx]

train_idx2, test_idx = next(msss.split(train_pool, multi[train_idx]))
train_ex = [train_pool[i] for i in train_idx2]
test_ex  = [train_pool[i] for i in test_idx]

ds_raw = DatasetDict(
    train=Dataset.from_list(train_ex),
    dev  = Dataset.from_list(dev_ex),
    test = Dataset.from_list(test_ex),
)

# ═════ 3. tokenise ══════════════════════════════════════════════════════
tok = AutoTokenizer.from_pretrained(MODEL_ID)
def tok_fn(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = batch["labels"]
    return enc

ds = ds_raw.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tok, return_tensors="pt")

# ═════ 4. model & weighted BCE ══════════════════════════════════════════
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=L,
    problem_type="multi_label_classification",
    id2label={i:l for i,l in enumerate(types_vocab)},
    label2id=lab2id,
).to(device)

freq = collections.Counter(np.where(np.array(e["labels"]))[0][0] for e in examples)
pos_w = torch.tensor([np.sqrt((len(examples)-freq[i])/freq[i]) for i in range(L)],
                     dtype=torch.float32)

class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(device, dtype=torch.float32)
        outs = model(**{k:v.to(device) for k,v in inputs.items()})
        loss = F.binary_cross_entropy_with_logits(
            outs.logits, labels,
            pos_weight=pos_w.to(device, dtype=outs.logits.dtype)
        )
        return (loss, outs) if return_outputs else loss

def micro_f1(pred: EvalPrediction):
    probs = 1/(1+np.exp(-pred.predictions))
    preds = (probs >= 0.5).astype(int)
    return {"micro_f1": f1_score(pred.label_ids, preds,
                                 average="micro", zero_division=0)}

args_tr = TrainingArguments(
    output_dir=str(CKPT_DIR),
    per_device_train_batch_size=BS_TRAIN,
    per_device_eval_batch_size=BS_EVAL,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=True,
    seed=SEED,
    remove_unused_columns=False,
    report_to="none",
)

trainer = BCETrainer(
    model=model, args=args_tr,
    train_dataset=ds["train"], eval_dataset=ds["dev"],
    data_collator=collator, compute_metrics=micro_f1, tokenizer=tok)
trainer.train()

# ═════ 5. threshold sweep on dev ════════════════════════════════════════
dev_loader = torch.utils.data.DataLoader(ds["dev"], batch_size=BS_EVAL,
                                         collate_fn=collator)
dev_logits, dev_labels = [], []
model.eval()
with torch.no_grad():
    for batch in dev_loader:
        lbl = batch.pop("labels")
        log = model(**{k:v.to(device) for k,v in batch.items()}).logits.cpu()
        dev_logits.append(log); dev_labels.append(lbl)
dev_logits = torch.cat(dev_logits); dev_labels = torch.cat(dev_labels)

best_thr, best_f1 = 0, 0
for thr in THR_GRID:
    f1 = f1_score(dev_labels,
                  (torch.sigmoid(dev_logits) > thr).int(),
                  average="micro", zero_division=0)
    if f1 > best_f1: best_thr, best_f1 = thr, f1
print(f"[DEV] best micro-F1 {best_f1:.3f} at thr={best_thr:.2f}")

# ═════ 6. inference on official test ════════════════════════════════════
test_rows = json.loads(ORIG_TEST.read_text("utf-8"))
defs_master = json.loads(DEF_FILE.read_text("utf-8"))

def build_txt(term: str) -> str:
    return term if not USE_DEFINITIONS else f"[TERM] {term}\n[DEF] {defs_master.get(term,'')}"

submission, batch = [], 32
model.eval()
with torch.no_grad():
    for i in range(0, len(test_rows), batch):
        slice_rows = test_rows[i:i+batch]
        enc = tok([build_txt(r["term"]) for r in slice_rows],
                  padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        probs = torch.sigmoid(model(**enc).logits).cpu().numpy()
        for r, p in zip(slice_rows, probs):
            idx = [j for j, v in enumerate(p) if v >= best_thr][:TOP_K] or [int(p.argmax())]
            submission.append({"id": r["id"], "types": [types_vocab[j] for j in idx]})

Path(SUB_FILE).write_text(json.dumps(submission, indent=2))
print(f"[DONE] {len(submission)} test rows → {SUB_FILE}")
