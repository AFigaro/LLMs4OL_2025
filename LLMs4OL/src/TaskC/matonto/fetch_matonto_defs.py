"""
Collect one-sentence English definitions for MatOnto ontology types.

 • Re-uses definitions already present in SWEET’s wiki_leads_patched.json
 • Reports overlap (MatOnto ∩ SWEET)
 • Fetch order:
        1. Wikipedia REST “summary”      (lead paragraph)
        2. Wiktionary REST
        3. USGS Mineral Glossary         (JSON endpoint)
        4. NOAA NWS Glossary             (HTML scrape)
        5. ADS “Universal glossary”
        6. OpenAI synthetic definition   (only if OPENAI_API_KEY is set)
 • Saves/updates  src/TaskC/matonto_leads_patched.json   after *each* term
"""

import json, requests, re, html, urllib.parse, time, os, sys
from pathlib import Path
from typing import Optional, Callable
from functools import lru_cache
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt_syn FROM SOURCES"
# ───────────────────────── CONFIG ─────────────────────────────────────────
ROOT              = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
SWEET_CACHE       = ROOT / "src/TaskC/matonto/matonto_types_leads.json"
MATONTO_DIR       = ROOT / "2025/TaskC-TaxonomyDiscovery/MatOnto"
TRAIN_TYPES_TXT   = MATONTO_DIR / "train" / "matonto_train_types.txt"
# if you have a test list, add it here:
TEST_TYPES_TXT  = MATONTO_DIR / "test"  / "matonto_test_types.txt"

OUT_FILE          = ROOT / "src" / "TaskC" / "matonto" / "matonto_types_leads.json"
UA                = {"User-Agent": "MatOnto-DefinitionFetcher/1.0"}
TIMEOUT           = 2        # seconds per HTTP request
SLEEP             = 0.3      # politeness delay
FLAG              = "The term does not have Wikipedia article"  # legacy flag

# ───────────────────────── HELPERS ────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")
def clean(text: str) -> str:
    """Strip HTML tags & collapse whitespace."""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", text))).strip()

def safe_write(payload: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

# ───────────────────────── SOURCE FETCHERS ────────────────────────────────
def wikipedia_lead(term: str) -> Optional[str]:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        js = r.json()
        return clean(js.get("extract", "")) or None
    except Exception:
        return None

def wiktionary(term: str) -> Optional[str]:
    REST = "https://en.wiktionary.org/api/rest_v1/page/definition/{}"
    for v in (term, term.lower(), term.capitalize()):
        try:
            r = requests.get(REST.format(urllib.parse.quote(v)), headers=UA, timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            js = r.json().get("en", [])
            if js and js[0]["definitions"]:
                return clean(js[0]["definitions"][0]["definition"])
        except Exception:
            pass
    return None

def usgs(term: str) -> Optional[str]:
    url = f"https://mrdata.usgs.gov/metadata/mineral/{urllib.parse.quote(term.lower())}.json"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return clean(r.json().get("definition", "")) or None
    except Exception:
        return None

def noaa(term: str) -> Optional[str]:
    url = f"https://forecast.weather.gov/glossary.php?word={urllib.parse.quote(term)}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        m = re.search(r'<p class="glossaryProductDescription">(.*?)</p>', r.text, re.S)
        return clean(m.group(1)) if m else None
    except Exception:
        return None

def ads(term: str) -> Optional[str]:
    q = urllib.parse.quote(f'property:"glossary" title:"{term}"')
    url = f"https://ui.adsabs.harvard.edu/v1/search/query?q={q}&fl=abstract&rows=1"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        docs = r.json().get("response", {}).get("docs", [])
        return clean(docs[0]["abstract"]) if docs and docs[0].get("abstract") else None
    except Exception:
        return None

def synthetic_openai(term: str) -> Optional[str]:
        prompt = f"Give a one-sentence English definition of the term '{term}' used in materials science."
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2,
        )
        return res.choices[0].message.content.strip()


SOURCES: list[Callable[[str], Optional[str]]] = [
    wikipedia_lead,
    wiktionary,
    usgs,
    noaa,
    ads,
    synthetic_openai,
]

# ───────────────────────── MAIN FLOW ──────────────────────────────────────
def main():
    # --- MatOnto type list -------------------------------------------------
    mat_types = set(Path(TRAIN_TYPES_TXT).read_text(encoding="utf-8").splitlines())
    if TEST_TYPES_TXT.exists():  mat_types |= set(Path(TEST_TYPES_TXT).read_text("utf-8").splitlines())
    mat_types = {t.strip() for t in mat_types if t.strip()}
    print(f"[INFO] MatOnto distinct types  : {len(mat_types):,}")

    # --- Load SWEET cache --------------------------------------------------
    sweet_defs = json.loads(SWEET_CACHE.read_text(encoding="utf-8"))
    overlap = {t for t in mat_types if t in sweet_defs and sweet_defs[t].strip() and not sweet_defs[t].startswith(FLAG)}
    print(f"[INFO] Already defined via SWEET: {len(overlap):,}  ({len(overlap)/len(mat_types):.1%})")

    # --- Initialise output dict -------------------------------------------
    if OUT_FILE.exists():
        defs = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    else:
        defs = {}
    # carry over SWEET overlaps
    for t in overlap:
        defs.setdefault(t, sweet_defs[t])

    missing = [t for t in sorted(mat_types) if t not in defs or not defs[t].strip()]
    print(f"[INFO] Still missing definitions : {len(missing):,}")

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
            definition = FLAG  # mark as missing
        defs[term] = definition
        safe_write(defs, OUT_FILE)
        time.sleep(SLEEP)

    print(f"\n[DONE] Definitions saved to {OUT_FILE}")
    print(f"       Total with definitions : {sum(1 for v in defs.values() if v and not v.startswith(FLAG)):,}")

if __name__ == "__main__":
    main()
