# -------------------------------------------------------------------------
# Fill {term : one-sentence definition} for ALL terms in SWEET Task-B.
#
#   1. Looks at SWEET train + test JSON files.
#   2. Re-uses existing defs in
#        src/TaskB/sweet/matonto_term_definitions.json
#   3. Fetch order:
#        • Wikipedia REST summary
#        • Wiktionary REST
#        • USGS Mineral Glossary
#        • NOAA NWS Glossary
#        • ADS “Universal glossary”
#        • GPT synthetic (only if OPENAI_API_KEY is set)
#   4. Writes / updates the SWEET definitions file after EACH new term.
# -------------------------------------------------------------------------

import json, requests, re, html, urllib.parse, time, os
from pathlib import Path
from typing import Optional, Callable
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt FROM SOURCES"
# To run the script add original data files (2025) to LLMs4OL folder


# ───────── PATHS ────────────────────────────────────────────────────────
ROOT = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_JSON = ROOT / "2025/TaskB-TermTyping/SWEET/train/term_typing_train_data.json"
TEST_JSON  = ROOT / "2025/TaskB-TermTyping/SWEET/test/sweet_term_typing_test_data.json"

OUT_FILE = ROOT / "src/TaskB/sweet/sweet_term_definitions.json"

UA       = {"User-Agent": "SWEET-TermDefFetcher/1.0"}
TIMEOUT  = 2
SLEEP    = 0.3
FLAG     = "The term does not have Wikipedia article"

# ───────── UTILITIES ────────────────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")
def clean(txt: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", txt))).strip()

def safe_write(payload: dict):
    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, OUT_FILE)

# ───────── FETCHERS ─────────────────────────────────────────────────────
def wiki(term: str) -> Optional[str]:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
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
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200: return None
        return clean(r.json().get("definition", "")) or None
    except Exception: return None

def noaa(term: str) -> Optional[str]:
    try:
        url = f"https://forecast.weather.gov/glossary.php?word={urllib.parse.quote(term)}"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
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

def gpt(term: str) -> Optional[str]:
    prompt = f"Give a one-sentence English definition of the earth-science term '{term}'."
    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80, temperature=0.2)
    return res.choices[0].message.content.strip()


SOURCES: list[Callable[[str], Optional[str]]] = [wiki, wiktionary, usgs, noaa, ads, gpt]

# ───────── MAIN ─────────────────────────────────────────────────────────
def main():
    # collect SWEET terms
    terms = {obj["term"].strip()
             for path in (TRAIN_JSON, TEST_JSON)
             for obj in json.loads(path.read_text("utf-8"))}
    print(f"[INFO] SWEET terms total : {len(terms):,}")

    # existing definitions
    defs = json.loads(OUT_FILE.read_text("utf-8")) if OUT_FILE.exists() else {}

    missing = [t for t in sorted(terms) if t not in defs or not defs[t].strip()]
    print(f"[INFO] need to fetch      : {len(missing):,}")

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
        safe_write(defs)
        time.sleep(SLEEP)

    good = sum(1 for v in defs.values() if v and not v.startswith(FLAG))
    print(f"\n[DONE] {good:,} / {len(defs):,} terms have definitions.")
    print(f"      File updated → {OUT_FILE}")

if __name__ == "__main__":
    main()