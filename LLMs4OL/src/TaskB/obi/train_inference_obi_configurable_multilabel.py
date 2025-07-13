# -----------------------------------------------------------------------
# Multilabel term-typing for the OBI Task-B data set.
#
# Switch behaviour by editing ONLY the two booleans below:
#
#     USE_AUGMENTED   = True  → use train_augmented.json
#                      = False → use original train json
#
#     USE_DEFINITIONS = True  → prepend “[DEF] …” text for training + test
#                      = False → feed only the bare term
#
# -----------------------------------------------------------------------
USE_AUGMENTED   = True      # ← edit me
USE_DEFINITIONS = True      # ← edit me
# -----------------------------------------------------------------------

import json, random, collections, os, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score
from tqdm import tqdm

# ───── paths & constants ────────────────────────────────────────────────
ROOT          = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
ORIG_TRAIN    = ROOT / "2025/TaskB-TermTyping/OBI/train/term_typing_train_data.json"
ORIG_TEST     = ROOT / "2025/TaskB-TermTyping/OBI/test/obi_term_typing_test_data.json"

AUG_TRAIN     = ROOT / "src/TaskB/obi/train_augmented.json"
TEST_WITH_DEF = ROOT / "src/TaskB/obi/test_with_defs.json"
DEF_FILE      = ROOT / "src/TaskB/obi/obi_term_definitions.json"

RUN_TAG   = f"{'aug' if USE_AUGMENTED else 'orig'}_" \
            f"{'defs' if USE_DEFINITIONS else 'nodef'}"

CKPT_DIR  = ROOT / f"src/TaskB/obi/checkpoints/{RUN_TAG}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SUB_FILE  = ROOT / f"src/TaskB/obi/preds_{RUN_TAG}.json"

MODEL_ID      = "microsoft/deberta-v3-large"
SEED          = 42
DEV_FRAC      = 0.05
BS_TRAIN      = 4
BS_EVAL       = 2
EPOCHS        = 30
LR            = 1e-5
MAX_LEN       = 1024
THR_SCAN      = np.linspace(0.05, 0.5, 10)   # global-thr sweep

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───── 1. load rows ─────────────────────────────────────────────────────
if USE_AUGMENTED:
    rows = json.loads(AUG_TRAIN.read_text("utf-8"))
    # already exploded: one type per row
    triplets = [(r["term"], r.get("definition", ""), r["type"]) for r in rows]
else:
    orig = json.loads(ORIG_TRAIN.read_text("utf-8"))
    defs = json.loads(DEF_FILE.read_text("utf-8"))
    triplets = [(r["term"], defs.get(r["term"], ""), t)
                for r in orig for t in r["types"]]

types_vocab = sorted({t for _,_,t in triplets})
lab2id = {t:i for i,t in enumerate(types_vocab)}
L = len(types_vocab)

def build_row(term, definition, typ):
    txt = (f"[TERM] {term}\n[DEF] {definition}"
           if USE_DEFINITIONS else term)
    vec = [0]*L; vec[lab2id[typ]] = 1
    return {"text": txt, "labels": vec}

examples = [build_row(*tri) for tri in triplets]
random.shuffle(examples)

# ───── 2. stratified split (train/dev/test-in) ──────────────────────────
multi = torch.tensor([e["labels"] for e in examples])
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=DEV_FRAC, random_state=SEED)
train_idx, dev_idx = next(msss.split(examples, multi))
train_pool = [examples[i] for i in train_idx]
dev_ex     = [examples[i] for i in dev_idx]

train_idx2, test_idx = next(msss.split(train_pool, multi[train_idx]))
train_ex = [train_pool[i] for i in train_idx2]
test_ex  = [train_pool[i] for i in test_idx]

ds_raw = DatasetDict(
    train = Dataset.from_list(train_ex),
    dev   = Dataset.from_list(dev_ex),
    test  = Dataset.from_list(test_ex),
)

# ───── 3. tokenise ──────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_ID)
def tok_fn(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = batch["labels"]
    return enc

ds = ds_raw.map(tok_fn, batched=True, remove_columns=["text"])
collate = DataCollatorWithPadding(tok, return_tensors="pt")

# ───── 4. model & trainer ───────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=L,
    problem_type="multi_label_classification",
    id2label={i:l for l,i in lab2id.items()},
    label2id=lab2id,
).to(device)

# imbalance weights
cnt = collections.Counter(np.where(np.array(e["labels"]))[0][0] for e in examples)
pos_w = torch.tensor([(len(examples)-cnt[i])/cnt[i] for i in range(L)], dtype=torch.float32)

class BCETrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, **_
    ):
        # move + cast labels to the same dtype as logits
        labels = inputs.pop("labels").to(device)          # on GPU
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        
        labels = labels.to(dtype=outputs.logits.dtype)    # ← add this
        pw     = pos_w.to(device, dtype=outputs.logits.dtype)  # ← and this

        loss = F.binary_cross_entropy_with_logits(
            outputs.logits, labels, pos_weight=pw
        )
        return (loss, outputs) if return_outputs else loss

def micro_f1(pred: EvalPrediction):
    probs = 1/(1+np.exp(-pred.predictions))
    preds = (probs >= 0.5).astype(int)
    return {"micro_f1": f1_score(pred.label_ids, preds,
                                 average="micro", zero_division=0)}

args_tr = TrainingArguments(
    output_dir=str(CKPT_DIR / f"{'aug' if USE_AUGMENTED else 'orig'}_{'defs' if USE_DEFINITIONS else 'nodef'}"),
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

trainer = BCETrainer(model=model, args=args_tr,
                     train_dataset=ds["train"], eval_dataset=ds["dev"],
                     data_collator=collate, compute_metrics=micro_f1)
trainer.train()

# ───── 5. global-thr sweep on dev ───────────────────────────────────────
dev_loader = torch.utils.data.DataLoader(ds["dev"], batch_size=BS_EVAL, collate_fn=collate)
dev_logits, dev_labels = [], []
with torch.no_grad():
    for batch in dev_loader:
        lbl = batch.pop("labels")
        dev_labels.append(lbl)
        dev_logits.append(model(**{k:v.to(device) for k,v in batch.items()}).logits.cpu())
dev_logits = torch.cat(dev_logits); dev_labels = torch.cat(dev_labels)

best_thr, best_f1 = 0.0, 0.0
for thr in THR_SCAN:
    f1 = f1_score(dev_labels, (torch.sigmoid(dev_logits) > thr).int(),
                  average="micro", zero_division=0)
    if f1 > best_f1:
        best_thr, best_f1 = thr, f1
print(f"[DEV] best micro-F1 {best_f1:.3f} @ thr {best_thr:.2f}")

# ───── 6. inference on official test set ────────────────────────────────
test_rows = json.loads(ORIG_TEST.read_text("utf-8"))
defs_test = {d["term"]: d["definition"] for d in
             (json.loads(TEST_WITH_DEF.read_text("utf-8")) if USE_DEFINITIONS else [])}

def txt(term):
    return f"[TERM] {term}\n[DEF] {defs_test.get(term,'')}" if USE_DEFINITIONS else term

model.eval(); submission=[]
with torch.no_grad():
    for i in range(0, len(test_rows), 32):
        slice_rows = test_rows[i:i+32]
        enc = tok([txt(r["term"]) for r in slice_rows],
                  padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        probs = torch.sigmoid(model(**enc).logits).cpu().numpy()
        for r,p in zip(slice_rows, probs):
            idx = [i for i,v in enumerate(p) if v>=best_thr][:3] or [int(p.argmax())]
            submission.append({"id":r["id"],"types":[types_vocab[i] for i in idx]})

Path(SUB_FILE).write_text(json.dumps(submission, indent=2))
print(f"[DONE]   {len(submission)} test rows  → {SUB_FILE}")