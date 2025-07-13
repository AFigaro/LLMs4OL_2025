"""
Collect one-sentence English definitions for **OBI** (Ontology for Biomedical Investigations).

• Fetch order:
      1. Wikipedia REST “summary”
      2. Wiktionary REST
      3. NCBI MeSH via E-utilities
      4. Merriam-Webster Medical (scrape)          ← no API key needed
      5. OpenAI synthetic definition (if OPENAI_API_KEY is set)
• Saves/updates  src/TaskC/obi_leads_patched.json   after *each* term
"""
from __future__ import annotations
import json, re, html, urllib.parse, os, time, requests
from pathlib import Path
from typing import Callable, Optional
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt_syn FROM SOURCES"

# ───────── CONFIG ─────────────────────────────────────────────────────────
ROOT              = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
SWEET_CACHE       = ROOT / "src" / "TaskC" / "obi_types_leads.json"

OBI_DIR           = ROOT / "2025/TaskC-TaxonomyDiscovery/OBI"
TRAIN_TYPES_TXT   = OBI_DIR / "train" / "obi_train_types.txt"
TEST_TYPES_TXT    = OBI_DIR / "test"  / "obi_test_types.txt"

OUT_FILE          = ROOT / "src" / "TaskC" / "obi" / "obi_leads_patched.json"

UA                = {"User-Agent": "OBI-DefinitionFetcher/1.0"}
TIMEOUT           = 3      # seconds per HTTP request
SLEEP             = 0.3    # polite delay
FLAG              = "No open-source definition found"


# ───────── HELPERS ────────────────────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")
def clean(text: str) -> str:
    """Strip HTML tags & collapse whitespace."""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", text))).strip()

def safe_write(payload: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

# ───────── SOURCE FETCHERS ────────────────────────────────────────────────
def wikipedia_lead(term: str) -> Optional[str]:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return clean(r.json().get("extract", "")) or None
    except Exception:
        return None

def wiktionary(term: str) -> Optional[str]:
    REST = "https://en.wiktionary.org/api/rest_v1/page/definition/{}"
    for v in (term, term.lower(), term.capitalize()):
        try:
            r = requests.get(REST.format(urllib.parse.quote(v)), headers=UA, timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json().get("en", [])
            if data and data[0]["definitions"]:
                return clean(data[0]["definitions"][0]["definition"])
        except Exception:
            pass
    return None

def mesh_definition(term: str) -> Optional[str]:
    """
    Query NCBI MeSH via E-utilities. Returns scope note if found.
    """
    try:
        esearch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=mesh&retmode=json&term={urllib.parse.quote(term)}[mh]"
        )
        r = requests.get(esearch, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        ids = r.json()["esearchresult"]["idlist"]
        if not ids:
            return None
        uid = ids[0]
        esum = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=mesh&retmode=json&id={uid}"
        )
        s = requests.get(esum, headers=UA, timeout=TIMEOUT).json()
        note = s["result"][uid].get("ds_meshscope_note")  # scope note field
        return clean(note) if note else None
    except Exception:
        return None

def merriam_medical(term: str) -> Optional[str]:
    """Scrape first definition line from Merriam-Webster Medical dictionary."""
    url = f"https://www.merriam-webster.com/medical/{urllib.parse.quote(term.replace(' ','-'))}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        m = re.search(r'<p class="definition-inner-item.*?">(.*?)</p>', r.text, re.S)
        return clean(m.group(1)) if m else None
    except Exception:
        return None

def synthetic_openai(term: str) -> Optional[str]:
    prompt = f"Provide a one-sentence English definition of the biomedical term or entity '{term}'."
    res = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.2,
    )
    return res.choices[0].message.content.strip()


SOURCES: list[Callable[[str], Optional[str]]] = [
    wikipedia_lead,
    wiktionary,
    mesh_definition,
    merriam_medical,
    synthetic_openai,
]

# ───────── MAIN FLOW ──────────────────────────────────────────────────────
def main():
    # ---------- OBI type list --------------------------------------------
    obi_types = set(TRAIN_TYPES_TXT.read_text("utf-8").splitlines())
    if TEST_TYPES_TXT.exists():
        obi_types |= set(TEST_TYPES_TXT.read_text("utf-8").splitlines())
    obi_types = {t.strip() for t in obi_types if t.strip()}
    print(f"[INFO] OBI distinct types      : {len(obi_types):,}")

    # ---------- load existing OBI cache ----------------------------------
    defs = json.loads(OUT_FILE.read_text("utf-8")) if OUT_FILE.exists() else {}
    for t in overlap:
        defs.setdefault(t, sweet_defs[t])

    missing = [t for t in sorted(obi_types) if t not in defs or not defs[t].strip()]
    print(f"[INFO] Still missing definitions: {len(missing):,}")

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

    have_defs = sum(1 for v in defs.values() if v and not v.startswith(FLAG))
    print(f"\n[DONE] Definitions saved → {OUT_FILE}")
    print(f"       Total with definitions : {have_defs:,}")

if __name__ == "__main__":
    main()
