#!/usr/bin/env python3
"""
────────────────────────────────────────────────────────────────────────
Augments MatOnto Task-B training rows into many surface variants and
writes `train_augmented.jsonl`.
Note that 

Variant sources
  • Unit word ↔ symbol  (m, kg, J …)
  • "per" ➜ "/" ; squared / cubed / power N ➜ ^2 ^3 ^-1 …
  • Simple tweaks  (plural, case, hyphen/space swap)
  • Wiktionary short synonyms       (cached)
  • GPT-4o short synonyms (≤3 words, cached, optional)

All external calls are cached under `_cache/` so the script is restart-safe.
"""

import json, re, unicodedata, html, urllib.parse, requests, os, time
from pathlib import Path
from typing import Set, Dict
from tqdm import tqdm
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt_syn FROM SOURCES"

# ───────── PATHS ────────────────────────────────────────────────────────
ROOT   = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
SRC    = ROOT / "2025/TaskB-TermTyping/MatOnto/train/term_typing_train_data.json"
TEST_JSON    = ROOT / "2025/TaskB-TermTyping/MatOnto/test/matonto_term_typing_test_data.json"
OUT_DIR      = ROOT / "src/TaskB/matonto"
OUT_TRAIN    = OUT_DIR / "train_augmented.json"
OUT_TEST     = OUT_DIR / "test_with_defs.json"

CACHE  = ROOT / "src/TaskB/matonto/_cache"
CACHE.mkdir(parents=True, exist_ok=True)
WIK_CACHE = CACHE / "wik_syn.json"
GPT_CACHE = CACHE / "gpt_syn.json"

# ───────── UNIT REPLACEMENTS (caret notation) ───────────────────────────
WORD2SYM = {
    "meter":"m","metre":"m","second":"s","kilogram":"kg","gram":"g",
    "ampere":"A","kelvin":"K","candela":"cd","mole":"mol","pascal":"Pa",
    "joule":"J","watt":"W","newton":"N","gray":"Gy","sievert":"Sv",
    "henry":"H","farad":"F","weber":"Wb","volt":"V","ohm":"Ω","tesla":"T"
}

POW_RE = re.compile(r"\b(squared|cubed|power\s*(-?\d+))\b", re.I)
def _word2pow(match):
    w = match.group(1).lower()
    if w.startswith("squared"):
        return "^2"
    if w.startswith("cubed"):
        return "^3"
    return f"^{match.group(2)}"

def unit_variants(term: str) -> Set[str]:
    t = term.lower()
    repl = t
    for w, s in WORD2SYM.items():
        repl = re.sub(rf"\b{w}\b", s, repl)
    repl = repl.replace(" per ", "/")
    repl = POW_RE.sub(_word2pow, repl)       # caret exponents
    dot  = repl.replace(" ", "·")            # middle-dot variant
    return {repl.strip(), dot.strip()} - {t}

# ───────── SIMPLE TWEAKS ────────────────────────────────────────────────
def simple_tweaks(term: str) -> Set[str]:
    out = {term}
    if not term.endswith("s"):
        out.add(term + "s")
    out.add(term.lower())
    out.add(term.title())
    out.add(term.replace("-", " "))
    out.add(term.replace(" ", "-"))
    return out - {term}

# ───────── WIKTIONARY SYNONYMS (cached) ─────────────────────────────────
REST_DEF = "https://en.wiktionary.org/api/rest_v1/page/definition/{}"
SPLIT_RE = re.compile(r",|;|/| or ")

wik_syn: Dict[str, list] = json.loads(WIK_CACHE.read_text()) if WIK_CACHE.exists() else {}
def wiktionary_synonyms(term: str) -> Set[str]:
    if term in wik_syn:
        return set(wik_syn[term])

    syns: Set[str] = set()
    try:
        url = REST_DEF.format(urllib.parse.quote(term))
        js  = requests.get(url, timeout=3,
                           headers={"User-Agent":"MatOntoAug/1.0"}).json()
        for block in js.get("en", []):
            for item in block.get("synonyms", []):
                txt = html.unescape(item).lower()
                for part in SPLIT_RE.split(txt):
                    tok = part.strip(" :—–").replace("’", "'")
                    if 0 < len(tok.split()) <= 3:
                        syns.add(tok)
    except Exception:
        pass

    wik_syn[term] = sorted(syns)
    WIK_CACHE.write_text(json.dumps(wik_syn, indent=2, ensure_ascii=False))
    time.sleep(0.2)
    return syns

# ───────── GPT-4o SYNONYMS (cached) ─────────────────────────────────────
gpt_syn: Dict[str, list] = json.loads(GPT_CACHE.read_text()) if GPT_CACHE.exists() else {}
def gpt_synonyms(term: str) -> Set[str]:
    if term in gpt_syn:
        return set(gpt_syn[term])
    if not openai.api_key:
        gpt_syn[term] = []
        return set()

    prompt = (
        f"Provide up to 3 concise English synonyms (≤ 3 words each) for the "
        f"scientific term “{term}”. Return them comma-separated, no extras."
    )
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=40, temperature=0.6,
        )
        txt = res.choices[0].message.content.lower()
        syns = {s.strip() for s in SPLIT_RE.split(txt) if 0 < len(s.split()) <= 3}
    except Exception:
        syns = set()

    gpt_syn[term] = sorted(syns)
    GPT_CACHE.write_text(json.dumps(gpt_syn, indent=2, ensure_ascii=False))
    time.sleep(0.3)
    return syns

# ───────── MAIN AUGMENTATION ────────────────────────────────────────────
train = json.loads(SRC.read_text("utf-8"))
augmented, seen = [], set()

for row in tqdm(train, desc="augment"):
    term = row["term"].strip()
    typ  = row["types"][0]                 # MatOnto rows have single type

    surfaces: Set[str] = {term}
    surfaces |= unit_variants(term)
    surfaces |= simple_tweaks(term)
    surfaces |= wiktionary_synonyms(term)
    surfaces |= gpt_synonyms(term)

    for surf in surfaces:
        s_norm = unicodedata.normalize("NFC", surf).strip()
        if not s_norm:
            continue
        key = (s_norm.lower(), typ)
        if key in seen:
            continue
        augmented.append({"text": s_norm, "label": typ})
        seen.add(key)

print(f"[INFO] original rows       : {len(train):,}")
print(f"[INFO] augmented examples  : {len(augmented):,}")

with DEST.open("w", encoding="utf-8") as f:
    for ex in augmented:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"[DONE] wrote {DEST}")

# ───────── BUILD TEST WITH DEFINTIONS ────────────────────────────────────────────
print("[STEP] preparing test file with definitions …")
test_out = [
    {
        "term": r["term"].strip(),
        "definition": definitions.get(r["term"].strip(), ""),
    }
    for r in tqdm(test_raw)
]

OUT_TEST.write_text(json.dumps(test_out, ensure_ascii=False, indent=2), "utf-8")
print(f"[DONE] wrote {OUT_TEST}")
