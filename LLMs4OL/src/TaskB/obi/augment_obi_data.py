# ----------------------------------------------------------------------
# Build an *augmented* training set for OBI Task-B and attach definitions
# to the test set.
#
#  • Adds surface variants:
#       ─ organisation aliases / acronyms
#       ─ clinical stage / grade rewrites
#       ─ unit / symbol flips (µ, °, /)
#       ─ acronym ↔ expansion table
#       ─ chemical formula ↔ common name (same helper as SWEET)
#       ─ Wiktionary short synonyms
#       ─ GPT-generated synonyms (≤3 words, optional)
#  • Rephrases each definition up to 3× with GPT (optional)
#  • Caches every LLM call so the script is restart-safe.
# ----------------------------------------------------------------------

import json, re, unicodedata, html, urllib.parse, requests, time, os
from pathlib import Path
from collections import Counter
from typing import Set, List, Dict, Tuple
from tqdm import tqdm
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt FROM SOURCES"
# ── PATHS ───────────────────────────────────────────────────────────────
ROOT         = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_JSON   = ROOT / "2025/TaskB-TermTyping/OBI/train/term_typing_train_data.json"
TEST_JSON    = ROOT / "2025/TaskB-TermTyping/OBI/test/obi_term_typing_test_data.json"
DEF_FILE     = ROOT / "src/TaskB/obi/obi_term_definitions.json"

OUT_DIR      = ROOT / "src/TaskB/obi"
OUT_TRAIN    = OUT_DIR / "train_augmented.json"
OUT_TEST     = OUT_DIR / "test_with_defs.json"

SYN_CACHE    = OUT_DIR / "llm_syn_cache.json"
DEF_CACHE    = OUT_DIR / "llm_def_cache.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

UA       = {"User-Agent": "OBI-Augment/1.0"}
TIMEOUT  = 3
SLEEP    = 0.15

# ════════════════════════════════════════════════════════════════════════
# 1.  Helper: chemical formula ↔ common name  (reuse SWEET logic)
# ════════════════════════════════════════════════════════════════════════
CHEM_MAP = {
    "h2o2":"hydrogen peroxide", "h2o":"water", "co2":"carbon dioxide",
    "ch3cooh":"acetic acid",    "ch4":"methane", "hno3":"nitric acid",
}
# add reverse mapping
for k, v in list(CHEM_MAP.items()):
    CHEM_MAP[v.lower()] = k

SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
DOTS = re.compile(r"[·•∙]")
def _norm_formula(s: str) -> str:
    return DOTS.sub("", unicodedata.normalize("NFKD", s).translate(SUBS)).lower().replace(" ", "")

def chem_variant(term: str) -> Set[str]:
    key = _norm_formula(term)
    return {CHEM_MAP[key]} if key in CHEM_MAP else set()

# ════════════════════════════════════════════════════════════════════════
# 2.  Organisation aliases
# ════════════════════════════════════════════════════════════════════════
ORG_SUFFIXES = [r",?\s+(inc\.?|corp\.?|corporation|biosciences|scientific|technologies)$"]
ACRONYM_PAREN = re.compile(r"\s+\(([^)]+)\)$")

def org_variants(term: str) -> Set[str]:
    t = term.strip()
    out = set()

    base = t
    for suf in ORG_SUFFIXES:
        base = re.sub(suf, "", base, flags=re.I)
    if base != t:
        out.add(base.strip())

    # acronym in parentheses
    m = ACRONYM_PAREN.search(t)
    if m:
        out.add(m.group(1).strip())

    # static expansions
    if len(t) <= 5 and t.isupper():
        EXPAND = {"BD": "Becton Dickinson", "AI": "Advanced Instruments", "ABI": "Applied Biosystems"}
        if t in EXPAND:
            out.add(EXPAND[t])
    else:
        REVERSE = {"becton dickinson": "BD", "advanced instruments": "AI", "applied biosystems": "ABI"}
        if t.lower() in REVERSE:
            out.add(REVERSE[t.lower()])
    return out - {t}

# ════════════════════════════════════════════════════════════════════════
# 3.  Clinical stage / grade patterns
# ════════════════════════════════════════════════════════════════════════
ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4"}
ROM_PAT = re.compile(r"\b(" + "|".join(ROMAN) + r")\b")

def stage_variants(term: str) -> Set[str]:
    out = set()
    base = term.strip()

    # roman → arabic
    if any(r in base for r in ROMAN):
        out.add(ROM_PAT.sub(lambda m: ROMAN[m.group(1)], base))

    # arabic → roman
    if any(d in base for d in ROMAN.values()):
        rev = {v: k for k, v in ROMAN.items()}
        out.add(re.sub(r"\b([1-4])\b", lambda m: rev[m.group(1)], base))

    # drop parenthetical site / system
    dropped = re.sub(r"\s*\([^)]*\)$", "", base).strip()
    if dropped != base:
        out.add(dropped)

    return out - {base}

# ════════════════════════════════════════════════════════════════════════
# 4.  Unit & symbol flips
# ════════════════════════════════════════════════════════════════════════
UNIT_MAP = {
    "liter": "l", "litre": "l", "second": "s", "gram": "g",
    "mole": "mol", "meter": "m", "metre": "m",
    "microliter": "µl", "microlitre": "µl", "microgram": "µg",
}
def unit_variants_obi(term: str) -> Set[str]:
    t = term.lower()
    out = set()
    for w, s in UNIT_MAP.items():
        if w in t:
            out.add(t.replace(w, s))
    if t.startswith("micro"):
        out.add("µ" + t[5:])
    if " per " in t:
        out.add(t.replace(" per ", "/"))
    if "degree celsius" in t:
        out.add(t.replace("degree celsius", "°c"))
    if "degree fahrenheit" in t:
        out.add(t.replace("degree fahrenheit", "°f"))
    return {o.strip() for o in out if o.strip()} - {term.lower()}

# ════════════════════════════════════════════════════════════════════════
# 5.  Acronym ↔ expansion table
# ════════════════════════════════════════════════════════════════════════
ACRO_MAP = {
    "ACS": "acute coronary syndrome",
    "FCS": "fetal calf serum",
    "OWL": "web ontology language",
    "XML": "extensible markup language",
    "RDF": "resource description framework",
}
def acronym_variants(term: str) -> Set[str]:
    t = term.strip()
    out = set()
    if t.upper() in ACRO_MAP:
        out.add(ACRO_MAP[t.upper()])
    else:
        for k, v in ACRO_MAP.items():
            if t.lower() == v:
                out.add(k)
    return out - {t}

# ════════════════════════════════════════════════════════════════════════
# 6.  Wiktionary short synonyms
# ════════════════════════════════════════════════════════════════════════
WIK_REST = "https://en.wiktionary.org/api/rest_v1/page/definition/{}"
SPLIT_RE = re.compile(r",|;|/| or ")

_wik_cache: Dict[str, Set[str]] = {}
def wiktionary_synonyms(term: str) -> Set[str]:
    if term in _wik_cache:
        return _wik_cache[term]

    syns: Set[str] = set()
    try:
        url = WIK_REST.format(urllib.parse.quote(term))
        js  = requests.get(url, headers=UA, timeout=TIMEOUT).json()
        for block in js.get("en", []):
            for item in block.get("synonyms", []):
                txt = html.unescape(item).strip().lower()
                for part in SPLIT_RE.split(txt):
                    tok = part.strip(" :—–").replace("’", "'")
                    if 0 < len(tok.split()) <= 3:
                        syns.add(tok)
        time.sleep(SLEEP)
    except Exception:
        pass

    _wik_cache[term] = syns
    return syns

# ════════════════════════════════════════════════════════════════════════
# 7.  GPT helpers + caches
# ════════════════════════════════════════════════════════════════════════
llm_syn = json.loads(SYN_CACHE.read_text()) if SYN_CACHE.exists() else {}
llm_def = json.loads(DEF_CACHE.read_text()) if DEF_CACHE.exists() else {}

def save_cache():
    SYN_CACHE.write_text(json.dumps(llm_syn, ensure_ascii=False, indent=2))
    DEF_CACHE.write_text(json.dumps(llm_def, ensure_ascii=False, indent=2))

def llm_synonyms(term: str) -> Set[str]:
    if term in llm_syn:
        return set(llm_syn[term])
    if not openai.api_key:
        llm_syn[term] = []
        return set()

    prompt = (
        f"List up to 3 short English synonyms (≤ 3 words) for the biomedical term '{term}'. "
        "Return only the synonyms, comma-separated."
    )
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
        )
        txt = res.choices[0].message.content.lower()
        syns = {t.strip() for t in re.split(r",|\n", txt) if 0 < len(t.split()) <= 3}
    except Exception:
        syns = set()

    llm_syn[term] = sorted(syns)
    save_cache()
    time.sleep(0.3)
    return syns

# ════════════════════════════════════════════════════════════════════════
# 8.  Load raw data & defs
# ════════════════════════════════════════════════════════════════════════
train_raw = json.loads(TRAIN_JSON.read_text("utf-8"))
test_raw  = json.loads(TEST_JSON.read_text("utf-8"))
definitions = json.loads(DEF_FILE.read_text("utf-8"))

def defs_of(term: str) -> List[str]:
    base = definitions.get(term, "").strip()
    parts = re.split(r"[;\n|]+", base)
    defs = [p.strip() for p in parts if p.strip()]
    out = []
    for d in defs or [""]:
        out.append(d)
    return sorted({x for x in out if x})

# ════════════════════════════════════════════════════════════════════════
# 9.  Build augmented TRAIN
# ════════════════════════════════════════════════════════════════════════
print("[STEP] augmenting OBI train set …")
aug, seen = [], set()

for row in tqdm(train_raw):
    orig = row["term"].strip()
    surfaces = {orig}
    surfaces.update(org_variants(orig))
    surfaces.update(stage_variants(orig))
    surfaces.update(unit_variants_obi(orig))
    surfaces.update(acronym_variants(orig))
    surfaces.update(chem_variant(orig))
    surfaces.update(wiktionary_synonyms(orig))
    surfaces.update(llm_synonyms(orig))

    for typ in row["types"]:
        for surf in surfaces:
            for d in defs_of(surf if surf != orig else orig):
                key = (surf.lower(), d, typ)
                if key in seen:
                    continue
                aug.append({"term": surf, "definition": d, "type": typ})
                seen.add(key)

print(f"[INFO] original pairs : {len(train_raw):,}")
print(f"[INFO] augmented pairs: {len(aug):,}")

OUT_TRAIN.write_text(json.dumps(aug, ensure_ascii=False, indent=2), "utf-8")
print(f"[DONE] wrote {OUT_TRAIN}")

# ════════════════════════════════════════════════════════════════════════
# 10.  Build TEST with definitions
# ════════════════════════════════════════════════════════════════════════
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
