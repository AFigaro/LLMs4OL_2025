#!/usr/bin/env python3
# collect_obi_term_defs.py  ── ontology-free version
# -------------------------------------------------------------------------
# 1. Reads OBI train & test JSON, accumulates unique terms.
# 2. Tries multiple NON-ontology biomedical sources to get a one-sentence
#    definition for each term.
# 3. Writes / updates obi_term_definitions.json incrementally.
# -------------------------------------------------------------------------

import json, os, re, time, html, urllib.parse, requests, textwrap
from pathlib import Path
from typing import Optional, Callable
import openai

openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt FROM SOURCES"
ROOT = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_JSON = ROOT / "2025/TaskB-TermTyping/OBI/train/term_typing_train_data.json"
TEST_JSON  = ROOT / "2025/TaskB-TermTyping/OBI/test/obi_term_typing_test_data.json"
OUT_FILE   = ROOT / "src/TaskB/obi/obi_term_definitions.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

UA      = {"User-Agent": "OBI-TermDefFetcher/1.0"}
TIMEOUT = 3
SLEEP   = 0.35
FLAG    = "No definition found in non-ontology sources"

# ── helpers ──────────────────────────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", text))).strip()

def safe_write(payload: dict):
    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(tmp, OUT_FILE)

# ── fetchers (NON-ontology) ─────────────────────────────────────────────
def mesh(term: str) -> Optional[str]:
    """MeSH scope note via NIH ClinicalTables."""
    try:
        url = f"https://clinicaltables.nlm.nih.gov/api/mesh_terms/v3/search?terms={urllib.parse.quote(term)}"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data[0] == 0 or not data[3]:
            return None
        return clean(data[3][0])
    except Exception:
        return None

def pubchem(term: str) -> Optional[str]:
    """PubChem compound summary."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(term)}/description/JSON"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        info = r.json().get("InformationList", {}).get("Information", [])
        if info and "Description" in info[0]:
            return clean(info[0]["Description"][0])
        return None
    except Exception:
        return None

def uniprot(term: str) -> Optional[str]:
    """UniProt protein ‘function’ line (good for enzymes, antibodies)."""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?fields=accession%2Cfunction&query={urllib.parse.quote(term)}&size=1"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if results and results[0]["comments"]:
            # take the first sentence of the first function comment
            func = results[0]["comments"][0]["texts"][0]["value"]
            return clean(func.split(".")[0] + ".")
        return None
    except Exception:
        return None

def wiki(term: str) -> Optional[str]:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return clean(r.json().get("extract", "")) or None
    except Exception:
        return None

def wiktionary(term: str) -> Optional[str]:
    try:
        url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{urllib.parse.quote(term)}"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        blocks = r.json().get("en", [])
        if blocks and blocks[0]["definitions"]:
            return clean(blocks[0]["definitions"][0]["definition"])
        return None
    except Exception:
        return None

def gpt(term: str) -> Optional[str]:

        import openai
        prompt = textwrap.dedent(f"""
            Provide a concise single-sentence biomedical definition of "{term}".
            Avoid any ontology jargon or references to IDs.
        """).strip()
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        return res.choices[0].message.content.strip()


SOURCES: list[Callable[[str], Optional[str]]] = [
    mesh, pubchem, uniprot, wiki, wiktionary, gpt
]

# ── main ────────────────────────────────────────────────────────────────
def main():
    terms = {obj["term"].strip()
             for path in (TRAIN_JSON, TEST_JSON)
             for obj in json.loads(path.read_text("utf-8"))}
    print(f"[INFO] unique OBI terms: {len(terms):,}")

    defs = json.loads(OUT_FILE.read_text("utf-8")) if OUT_FILE.exists() else {}

    missing = [t for t in sorted(terms) if not defs.get(t)]
    print(f"[INFO] definitions to fetch: {len(missing):,}")

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
    print(f"Definitions file → {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
