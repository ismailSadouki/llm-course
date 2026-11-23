"""Build the English+French training corpus from Europarl.

Source : Europarl (Koehn, 2005), as redistributed in the NLTK data repository
         (europarl_raw.zip, ~12 MB, downloaded from GitHub).
Output : data/processed/train_en.txt, train_fr.txt  (one sentence per line)
         data/processed/train_corpus.txt            (both, shuffled)

    python scripts/build_corpus.py                  # defaults: 3000 sentences per language
    python scripts/build_corpus.py --n 5000 --extra my_arabic.txt

--extra adds your own text file(s) (one sentence per line) to the mix, e.g. the
Arabic / Darija text you collected for Assignment 02's optional challenge.
You are responsible for the licence of any text you add.
"""
import argparse, io, pathlib, random, re, urllib.request, zipfile

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÝŒ«\"])")


def detokenize(s: str, lang: str) -> str:
    """Europarl in NLTK is pre-tokenized ("hope , in", "l ' ordre"); restore natural spacing
    so that the training text looks like the text the tokenizer will meet in real life."""
    s = re.sub(r"(\w)\s*['’]\s*(\w)", r"\1'\2", s)               # l ' ordre -> l'ordre, don ' t -> don't
    s = re.sub(r"\s+([,.%)\]])", r"\1", s)                         # no space before , . % ) ]
    s = re.sub(r"([(\[])\s+", r"\1", s)                            # no space after ( [
    if lang == "en":
        s = re.sub(r"\s+([;:!?])", r"\1", s)                       # English: no space before ; : ! ?
    return s


def sentences_from(zf: zipfile.ZipFile, folder: str, lang: str):
    names = sorted(n for n in zf.namelist() if n.startswith(f"europarl_raw/{folder}/") and not n.endswith("/"))
    for name in names:
        text = zf.read(name).decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("<"):          # SGML tags (<CHAPTER>, <SPEAKER>, <P>)
                continue
            for s in SPLIT.split(line):
                s = detokenize(" ".join(s.split()), lang)
                if 30 <= len(s) <= 220 and sum(c.isdigit() for c in s) <= 6:
                    yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="sentences per language")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[], help="extra text files (one sentence per line)")
    args = ap.parse_args()

    print("downloading Europarl (12 MB)...")
    zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()))
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    everything = []
    for lang, folder in [("en", "english"), ("fr", "french")]:
        pool = sorted(set(sentences_from(zf, folder, lang)))       # sorted -> reproducible
        rng.shuffle(pool)
        chosen = pool[: args.n]
        (OUT / f"train_{lang}.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
        print(f"{lang}: {len(pool)} candidates -> {len(chosen)} kept")
        everything += chosen
    for extra in args.extra:
        lines = [l.strip() for l in pathlib.Path(extra).read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"extra {extra}: {len(lines)} lines")
        everything += lines
    rng.shuffle(everything)
    (OUT / "train_corpus.txt").write_text("\n".join(everything) + "\n", encoding="utf-8")
    print("train_corpus.txt:", len(everything), "lines")


if __name__ == "__main__":
    main()
