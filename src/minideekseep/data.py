import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[NOISE]"]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def split_for(digest):
    value = int(digest[:8], 16) % 100
    return "validation" if value == 98 else "test" if value == 99 else "train"


def preview_rows(language):
    import time
    from urllib.parse import urlencode
    from urllib.request import urlopen

    offset = 0
    while True:
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {
                "dataset": "HuggingFaceFW/finewiki",
                "config": language,
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        for attempt in range(3):
            try:
                with urlopen(url, timeout=30) as response:
                    rows = json.load(response)["rows"]
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1)
        if not rows:
            return
        for row in rows:
            yield row["row"]
        offset += len(rows)


def collect(destination, source=None, documents=30000, revision=None, preview=False):
    """Stream bounded inputs; preserve source metadata without retaining an HF corpus locally."""
    destination = Path(destination)
    if documents < 1:
        raise ValueError("Document limit must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    corpus = destination / "documents.jsonl"
    if corpus.exists():
        raise FileExistsError(f"Refusing to replace existing corpus: {corpus}")
    seen = set()
    counts = {"train": 0, "validation": 0, "test": 0}
    from contextlib import ExitStack

    with ExitStack() as stack:
        out = stack.enter_context(corpus.with_suffix(".partial").open("w", encoding="utf-8"))
        if source:
            handle = stack.enter_context(Path(source).open(encoding="utf-8"))
            streams = [(None, (json.loads(line) for line in handle))]
            resolved_revision = "local:" + fingerprint(source)
        else:
            from huggingface_hub import HfApi

            if not preview:
                from datasets import load_dataset

            resolved_revision = (
                HfApi().dataset_info("HuggingFaceFW/finewiki", revision=revision).sha
            )
            if preview and revision:
                raise ValueError(
                    "The preview API reads current data; it cannot pin an old revision"
                )
            streams = [
                (
                    lang,
                    preview_rows(lang)
                    if preview
                    else load_dataset(
                        "HuggingFaceFW/finewiki",
                        lang,
                        split="train",
                        streaming=True,
                        revision=resolved_revision,
                    ),
                )
                for lang in ("zh", "en")
            ]
        for language, stream in streams:
            kept = 0
            limit = documents if language != "en" else max(100, documents // 10)
            for row in stream:
                text = normalize(row["text"])
                if len(text) < 64:
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                split = split_for(digest)
                record = dict(
                    text=text,
                    sha256=digest,
                    split=split,
                    language=language or row.get("language", "zh"),
                    id=row.get("id", digest),
                    url=row.get("url"),
                    title=row.get("title"),
                    revision=resolved_revision,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                kept += 1
                if kept >= limit:
                    break
        if preview and not source:
            if HfApi().dataset_info("HuggingFaceFW/finewiki").sha != resolved_revision:
                raise RuntimeError("Dataset changed during preview collection; use a new directory")
    corpus.with_suffix(".partial").replace(corpus)
    manifest = dict(
        source=str(source) if source else "HuggingFaceFW/finewiki",
        revision=resolved_revision,
        counts=counts,
        corpus_sha256=fingerprint(corpus),
        license="user-supplied" if source else "CC-BY-SA-4.0 / GFDL; Wikipedia contributors",
        transport="local" if source else "preview-current" if preview else "revision-pinned-stream",
    )
    (destination / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def train_tokenizer(directory, vocab_size=8192):
    directory = Path(directory)

    def training_texts():
        with (directory / "documents.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] == "train":
                    yield row["text"]

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(training_texts(), trainer=trainer)
    tokenizer.save(str(directory / "tokenizer.json"))
    return tokenizer


def encode_corpus(directory, tokenizer):
    directory = Path(directory)
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("uint16 shards require a vocabulary of at most 65536 tokens")
    offsets = {f"{s}_{lang}": [0] for s in ("train", "validation", "test") for lang in ("zh", "en")}
    counts = {k: 0 for k in offsets}
    from contextlib import ExitStack

    with ExitStack() as stack:
        handles = {
            key: stack.enter_context((directory / f"{key}.bin").open("wb")) for key in offsets
        }
        with (directory / "documents.jsonl").open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                language = row["language"]
                if language not in ("zh", "en"):
                    raise ValueError(f"Unsupported language {language}")
                key = f"{row['split']}_{language}"
                ids = [1] + tokenizer.encode(row["text"]).ids + [2]
                handles[key].write(np.asarray(ids, dtype="<u2").tobytes())
                offsets[key].append(offsets[key][-1] + len(ids))
                counts[key] += len(ids)
    for key, values in offsets.items():
        np.save(directory / f"{key}.offsets.npy", np.asarray(values, dtype=np.int64))
    manifest = dict(
        tokenizer_sha256=fingerprint(directory / "tokenizer.json"),
        vocab_size=tokenizer.get_vocab_size(),
        token_counts=counts,
        shard_sha256={key: fingerprint(directory / f"{key}.bin") for key in offsets},
        offsets_sha256={key: fingerprint(directory / f"{key}.offsets.npy") for key in offsets},
    )
    (directory / "encoded.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_corpus(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "encoded.json").read_text())
    if fingerprint(directory / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer file no longer matches the encoded corpus")
    for key, digest in manifest["shard_sha256"].items():
        if fingerprint(directory / f"{key}.bin") != digest:
            raise ValueError(f"Encoded shard was changed: {key}")
        offsets = np.load(directory / f"{key}.offsets.npy")
        if fingerprint(directory / f"{key}.offsets.npy") != manifest["offsets_sha256"][key]:
            raise ValueError(f"Document offsets were changed: {key}")
        count = (directory / f"{key}.bin").stat().st_size // 2
        if offsets[0] != 0 or offsets[-1] != count or (np.diff(offsets) <= 0).any():
            raise ValueError(f"Invalid document offsets: {key}")
    return manifest


class TokenStream:
    """A resumable sampler: each sequence stays inside one document, with padding ignored."""

    def __init__(self, directory, split="train", length=256, seed=42):
        self.directory = Path(directory)
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.data, self.offsets = {}, {}
        for lang in ("zh", "en"):
            path = self.directory / f"{split}_{lang}.bin"
            if path.exists() and path.stat().st_size:
                self.data[lang] = np.memmap(path, dtype="<u2", mode="r")
                self.offsets[lang] = np.load(self.directory / f"{split}_{lang}.offsets.npy")
        if not self.data:
            raise ValueError(f"No encoded documents in {split}")
        self.languages = list(self.data)
        self.english_credit = 0.0
        self.tokens = 0

    def next(self, device="cpu"):
        # Deficit scheduling balances real, unpadded tokens rather than document counts.
        language = "en" if self.english_credit > 0 and "en" in self.data else self.languages[0]
        if language == "en" and "zh" not in self.data:
            language = "en"
        boundaries = self.offsets[language]
        doc = int(self.rng.integers(len(boundaries) - 1))
        lo, hi = int(boundaries[doc]), int(boundaries[doc + 1])
        start = lo + int(self.rng.integers(max(1, hi - lo - self.length)))
        values = np.array(
            self.data[language][start : min(hi, start + self.length + 1)], dtype=np.int64
        )
        x = np.zeros(self.length, dtype=np.int64)
        y = np.full(self.length, -100, dtype=np.int64)
        n = len(values) - 1
        x[:n], y[:n] = values[:-1], values[1:]
        self.tokens += n
        self.english_credit += (0.05 - (language == "en")) * n
        return torch.from_numpy(x)[None].to(device), torch.from_numpy(y)[None].to(device)

    def state_dict(self):
        return dict(
            rng=self.rng.bit_generator.state,
            english_credit=self.english_credit,
            tokens=self.tokens,
            length=self.length,
        )

    def load_state_dict(self, state):
        if state["length"] != self.length:
            raise ValueError("Resume requires the original sequence length")
        self.rng.bit_generator.state = state["rng"]
        self.english_credit, self.tokens = state["english_credit"], state["tokens"]
