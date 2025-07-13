import json, re, html, unicodedata, urllib.parse, requests, time, os
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List
from tqdm import tqdm
import mwparserfromhell  
import openai               

# ───────── PATHS ────────────────────────────────────────────────────────
ROOT       = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_JSON = ROOT / "2025/TaskB-TermTyping/SWEET/train/term_typing_train_data.json"
TEST_JSON  = ROOT / "2025/TaskB-TermTyping/SWEET/test/sweet_term_typing_test_data.json"
DEF_FILE   = ROOT / "src/TaskB/sweet/sweet_term_definitions.json"

OUT_DIR    = ROOT / "src/TaskB/sweet"
OUT_TRAIN  = OUT_DIR / "train_augmented.json"
OUT_TEST   = OUT_DIR / "test_with_defs.json"
# LLM caches
SYN_CACHE = OUT_DIR / "llm_syn_cache.json"
DEF_CACHE = OUT_DIR / "llm_def_cache.json"

UA      = {"User-Agent": "SWEET-Augment/2.0"}
TIMEOUT = 2
SLEEP   = 0.05
# ========== 1. Wiktionary DET-synonyms (REST) ===========================
REST_DEF = "https://en.wiktionary.org/api/rest_v1/page/definition/{}"
SPLIT_RE = re.compile(r",|;|/| or ")

_wik_cache: Dict[str, Set[str]] = {}

def wiktionary_synonyms(term: str) -> Set[str]:
    if term in _wik_cache:
        return _wik_cache[term]

    syns: Set[str] = set()
    try:
        url = REST_DEF.format(urllib.parse.quote(term))
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

# ========== 2. Unit-style variants =====================================
WORD2SYM = { "meter":"m","metre":"m","second":"s","kilogram":"kg","gram":"g",
             "ampere":"a","kelvin":"k","candela":"cd","mole":"mol",
             "pascal":"pa","joule":"j","watt":"w","newton":"n","gray":"gy",
             "sievert":"sv","henry":"h","farad":"f","weber":"wb","volt":"v",
             "ohm":"ω","tesla":"t"}
POW_RE = re.compile(r"\b(squared|cubed|power\s*(-?\d+))\b", re.I)
SUPERS = str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def exp_utf(n:str)->str: return n.translate(SUPERS)

def unit_variants(term:str)->Set[str]:
    t = term.lower()
    if not any(w in t for w in WORD2SYM) and " per " not in t and "power" not in t:
        return set()
    repl = t
    for w,s in WORD2SYM.items():
        repl = re.sub(rf"\b{w}\b", s, repl)
    repl = repl.replace(" per ", "/")
    repl = POW_RE.sub(lambda m: exp_utf(m.group(2) or ("2" if "squared" in m.group(1).lower() else "3")), repl)
    dot  = repl.replace(" ", "·")
    return {repl.strip(), dot.strip()} - {term.lower()}

# ========== 3. Chemical swaps  (normalised) ============================
CHEM_MAP = {
    "h2":"hydrogen","o2":"oxygen","o3":"ozone",
    "co":"carbon monoxide","co2":"carbon dioxide",
    "no":"nitric oxide","no2":"nitrogen dioxide","n2o":"nitrous oxide",
    "h2o":"water","h2o2":"hydrogen peroxide",
    "hno3":"nitric acid","hno2":"nitrous acid","hno4":"peroxynitric acid",
    "hcl":"hydrogen chloride","hf":"hydrogen fluoride","hbr":"hydrogen bromide",
    "so2":"sulfur dioxide","sox":"sulfur oxides",
    "ch4":"methane","c2h4":"ethene","c2h6":"ethane",
    "c3h8":"propane","c3h6":"propene",
    "c6h6":"benzene","c7h8":"toluene","c8h10":"xylene",
    "ch3cooh":"acetic acid","ch2o":"formaldehyde",
    "(ch3)2s":"dimethyl sulfide","c2h6s":"ethyl mercaptan",
    "ccl2f2":"dichlorodifluoromethane","cf2clbr":"bromochlorodifluoromethane",
    "cbrf3":"bromotrifluoromethane","c2br2f4":"dibromotetrafluoroethane",
    "cbrf2":"chlorobromofluoromethane",
}
for k,v in list(CHEM_MAP.items()):
    CHEM_MAP[v.lower()] = k
SUBS=str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
DOTS=re.compile(r"[·•∙]")
def norm_formula(s:str)->str:
    return DOTS.sub("", unicodedata.normalize("NFKD",s).translate(SUBS)).lower().replace(" ","")
def chem_variant(term:str)->Set[str]:
    key=norm_formula(term)
    return {CHEM_MAP[key]} if key in CHEM_MAP else set()

# ========== 4. GPT-4o augmentation =====================================

# load / init caches
SYN_CACHE_FILE = SYN_CACHE
DEF_CACHE_FILE = DEF_CACHE
llm_syn = json.loads(SYN_CACHE_FILE.read_text()) if SYN_CACHE_FILE.exists() else {}
llm_def = json.loads(DEF_CACHE_FILE.read_text()) if DEF_CACHE_FILE.exists() else {}

def save_cache():
    SYN_CACHE_FILE.write_text(json.dumps(llm_syn, ensure_ascii=False, indent=2))
    DEF_CACHE_FILE.write_text(json.dumps(llm_def, ensure_ascii=False, indent=2))

def llm_synonyms(term:str)->Set[str]:
    if term in llm_syn:
        return set(llm_syn[term])
    if not openai.api_key:
        llm_syn[term]=[]
        return set()

    prompt = (f"List up to 3 short English synonyms (≤ 3 words) for the scientific term '{term}'. "
              "Return only the synonyms, comma-separated.")
    try:
        res=openai.ChatCompletion.create(model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=50,temperature=0.7)
        txt=res.choices[0].message.content.lower()
        syns={t.strip() for t in re.split(r",|\n",txt) if 0<len(t.split())<=3}
    except Exception:
        syns=set()
    llm_syn[term]=sorted(syns)
    save_cache()
    time.sleep(0.3)
    return syns

# ========== 5.  LOAD data & helpers =====================================
train_raw=json.loads(TRAIN_JSON.read_text("utf-8"))
test_raw=json.loads(TEST_JSON.read_text("utf-8"))
definitions=json.loads(DEF_FILE.read_text("utf-8"))

def defs_of(term:str)->List[str]:
    base=definitions.get(term,"").strip()
    parts=re.split(r"[;\n|]+",base)
    defs=[p.strip() for p in parts if p.strip()]
    out=[]
    for d in defs or [""]:
        out.append(d)

    return sorted({x for x in out if x})

# ========== 6.  Build augmented TRAIN ===================================
aug=[]
seen=set()
print("[STEP] augmenting train …")
for row in tqdm(train_raw):
    orig=row["term"].strip()
    surfaces={orig}
    surfaces.update(wiktionary_synonyms(orig))
    surfaces.update(unit_variants(orig))
    surfaces.update(chem_variant(orig))
    surfaces.update(llm_synonyms(orig))

    for typ in row["types"]:
        for surf in surfaces:
            for d in defs_of(surf if surf!=orig else orig):
                key=(surf.lower(),d,typ)
                if key in seen: continue
                aug.append({"term":surf,"definition":d,"type":typ})
                seen.add(key)

print(f"[INFO] original pairs : {len(train_raw):,}")
print(f"[INFO] augmented pairs: {len(aug):,}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRAIN.write_text(json.dumps(aug,ensure_ascii=False,indent=2),"utf-8")
print(f"[DONE] wrote {OUT_TRAIN}")

# ========== 7.  Build TEST =============================================
print("[STEP] preparing test …")
test_out=[{"term":r["term"].strip(),
           "definition":definitions.get(r["term"].strip(),"")} for r in tqdm(test_raw)]
OUT_TEST.write_text(json.dumps(test_out,ensure_ascii=False,indent=2),"utf-8")
print(f"[DONE] wrote {OUT_TEST}")