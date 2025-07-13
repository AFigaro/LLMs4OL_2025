#  • Writes output after *each* term (atomic temp → rename)
#  • Prints progress immediately (flush=True)
#  • Short 2-second HTTP timeouts to avoid long stalls
#  • Sources (in order):
#        1. Wiktionary REST
#        2. USGS Mineral Glossary
#        3. NOAA NWS Glossary
#        4. NASA ADS “Universal glossary”
#        5. GPT4o generated

import json, requests, re, html, urllib.parse, time, os
from pathlib import Path
import openai
openai.api_key = "PUT YOUR KEY HERE, OR REMOVE gpt_syn FROM SOURCES"
# ───────────────────────── CONFIG ────────────────────────────────────────
ROOT       = Path("/LLMs4OL/")
# To run the script add original data files (2025) to LLMs4OL folder
TRAIN_TYPES_TXT   = ROOT / "train" / "sweet_train_types.txt"
TEST_TYPES_TXT    = ROOT / "test"  / "sweet_test_types.txt"
FILE_OUT   = ROOT / "src/TaskC/sweet_types_lead.json"
FLAG       = "The term does not have Wikipedia article"
UA         = {"User-Agent": "SWEET-DefinitionFetcher/1.0"}

TAG_RE = re.compile(r"<[^>]+>")
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", text))).strip()

TIMEOUT = 2     # seconds per HTTP request
SLEEP   = 0.3   # polite delay after each term

# ── Source fetchers ───────────────────────────────────────────────────────
def from_wiktionary(term):
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

def from_usgs(term):
    url = f"https://mrdata.usgs.gov/metadata/mineral/{urllib.parse.quote(term.lower())}.json"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return clean(r.json().get("definition", "")) or None
    except Exception:
        return None

def from_noaa(term):
    url = f"https://forecast.weather.gov/glossary.php?word={urllib.parse.quote(term)}"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        m = re.search(r'<p class="glossaryProductDescription">(.*?)</p>', r.text, re.S)
        return clean(m.group(1)) if m else None
    except Exception:
        return None

def from_ads(term):
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

def synthetic_def(term):

    prompt = (f"Give a one-sentence English definition of the term "
              f"'{term}' for an earth-science ontology.")
    res = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        max_tokens=100, temperature=0.2
        )
    return res.choices[0].message.content.strip()
SOURCES = [from_wiktionary, from_usgs, from_noaa, from_ads, synthetic_def]

# ── Atomic write helper ───────────────────────────────────────────────────
def safe_write(payload: dict):
    tmp = FILE_OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, FILE_OUT)   # POSIX-atomic

# ── Main loop ─────────────────────────────────────────────────────────────
data = set(TRAIN_TYPES_TXT.read_text("utf-8").splitlines())
    if TEST_TYPES_TXT.exists():
        data |= set(TEST_TYPES_TXT.read_text("utf-8").splitlines())
    data = {t.strip() for t in data if t.strip()}
missing = [t for t, d in data.items() if d.strip().startswith(FLAG)]
print(f"Need definitions for {len(missing)} terms")

filled = 0
for idx, term in enumerate(missing, 1):
    # early progress print
    print(f"[{idx}/{len(missing)}] {term} …", flush=True)

    if not data[term].strip().startswith(FLAG):
        continue  # could already be filled by previous interrupted run

    definition = None
    for fetcher in SOURCES:
        definition = fetcher(term)
        if definition:
            break
    if definition:
        data[term] = definition
        filled += 1
        print(f"   ✔ filled via {fetcher.__name__}", flush=True)
    else:
        print("   ✖ still missing", flush=True)

    # save after every term
    safe_write(data)

    # polite delay
    time.sleep(SLEEP)

print(f"\nFilled {filled} new definitions; output saved to {FILE_OUT}")
