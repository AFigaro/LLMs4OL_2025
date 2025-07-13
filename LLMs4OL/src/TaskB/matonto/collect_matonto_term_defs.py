# -------------------------------------------------------------------------
# Build / update a JSON map  { term : "one-sentence definition" }  for all
# terms that appear in MatOnto Task-B (train + test).
#
#   • Fetch order (stop at first hit):
#         1. Wikipedia REST summary
#         2. Wiktionary REST
#         3. USGS Mineral Glossary
#         4. NOAA NWS Glossary
#         5. ADS “Universal glossary”
#         6. OpenAI synthetic (only if OPENAI_API_KEY is set)
#   • Saves / updates   src/TaskB/matonto/matonto_term_definitions.json
#     after every newly-filled term.
# -------------------------------------------------------------------------

import json, requests, re, html, urllib.parse, time, os
from pathlib import Path
from typing import Optional, Callable
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt_syn FROM SOURCES"

# ───────── PATHS ──────────────────────────────────────────────────────────
ROOT             = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_JSON       = ROOT / "2025/TaskB-TermTyping/MatOnto/train/term_typing_train_data.json"
TEST_JSON        = ROOT / "2025/TaskB-TermTyping/MatOnto/test/matonto_term_typing_test_data.json"

# optional: re-use generic term definitions you may already have
GENERIC_DEF_FILE = Path("/home/alatipov/LLMs4OL/src/TaskB/matonto/term_definitions.json")        # leave as-is if absent
OUT_FILE         = ROOT / "src/TaskB/matonto/matonto_term_definitions.json"

UA      = {"User-Agent": "MatOnto-TermDefFetcher/1.0"}
TIMEOUT = 2          # seconds per HTTP request
SLEEP   = 0.3        # polite delay
FLAG    = "The term does not have Wikipedia article"   # placeholder for “missing”

# ───────── UTILITIES ──────────────────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", text))).strip()

def safe_write(payload: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

# ───────── SOURCE FETCHERS (same 7 as before) ────────────────────────────
def wiki_summary(term: str) -> Optional[str]:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"
        r   = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200: return None
        return clean(r.json().get("extract", "")) or None
    except Exception: return None

def wiktionary(term: str) -> Optional[str]:
    REST = "https://en.wiktionary.org/api/rest_v1/page/definition/{}"
    for v in (term, term.lower(), term.capitalize()):
        try:
            r = requests.get(REST.format(urllib.parse.quote(v)), headers=UA, timeout=TIMEOUT)
            if r.status_code != 200: continue
            blocks = r.json().get("en", [])
            if blocks and blocks[0]["definitions"]:
                return clean(blocks[0]["definitions"][0]["definition"])
        except Exception: pass
    return None

def usgs(term: str) -> Optional[str]:
    try:
        url = f"https://mrdata.usgs.gov/metadata/mineral/{urllib.parse.quote(term.lower())}.json"
        r   = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200: return None
        return clean(r.json().get("definition", "")) or None
    except Exception: return None

def noaa(term: str) -> Optional[str]:
    try:
        url = f"https://forecast.weather.gov/glossary.php?word={urllib.parse.quote(term)}"
        r   = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200: return None
        m = re.search(r'<p class="glossaryProductDescription">(.*?)</p>', r.text, re.S)
        return clean(m.group(1)) if m else None
    except Exception: return None

def ads(term: str) -> Optional[str]:
    try:
        q   = urllib.parse.quote(f'property:"glossary" title:"{term}"')
        url = f"https://ui.adsabs.harvard.edu/v1/search/query?q={q}&fl=abstract&rows=1"
        r   = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200: return None
        docs = r.json().get("response", {}).get("docs", [])
        return clean(docs[0]["abstract"]) if docs and docs[0].get("abstract") else None
    except Exception: return None

def gpt_syn(term: str) -> Optional[str]:
    prompt = f"Provide a one-sentence English definition of the materials-science term '{term}'."
    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80, temperature=0.2,
    )
    return res.choices[0].message.content.strip()

SOURCES: list[Callable[[str], Optional[str]]] = [
    wiki_summary,
    wiktionary,
    usgs,
    noaa,
    ads,
    gpt_syn,
]

# ───────── MAIN ──────────────────────────────────────────────────────────
def main():
    # 1) Collect all terms (train + test) ----------------------------------
    all_terms = set()
    for path in (TRAIN_JSON, TEST_JSON):
        for obj in json.loads(path.read_text(encoding="utf-8")):
            all_terms.add(obj["term"].strip())
    print(f"[INFO] Unique terms (train+test) : {len(all_terms):,}")

    # 2) Load existing definition stores ----------------------------------
    defs = {}
    if OUT_FILE.exists():
        defs.update(json.loads(OUT_FILE.read_text(encoding="utf-8")))
    if GENERIC_DEF_FILE.exists():
        defs.update(json.loads(GENERIC_DEF_FILE.read_text("utf-8")))

    # 3) Determine missing -------------------------------------------------
    missing = [t for t in sorted(all_terms) if t not in defs or not defs[t].strip()]
    print(f"[INFO] Missing definitions        : {len(missing):,}")

    # 4) Fetch loop --------------------------------------------------------
    for idx, term in enumerate(missing, 1):
        print(f"[{idx}/{len(missing)}] {term} …", flush=True)
        definition = None
        for fetch in SOURCES:
            definition = fetch(term)
            if definition:
                print(f"    ✔ via {fetch.__name__}", flush=True)
                break
        if not definition:
            print("    ✖ all sources failed", flush=True)
            definition = FLAG

        defs[term] = definition
        safe_write(defs, OUT_FILE)
        time.sleep(SLEEP)

    good = sum(1 for v in defs.values() if v and not v.startswith(FLAG))
    print(f"\n[DONE] {good:,} / {len(defs):,} terms now have definitions.")
    print(f"      File → {OUT_FILE}")

if __name__ == "__main__":
    main()
