from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import CLIPTokenizer, CLIPTokenizerFast


TOKENIZER_ID = "openai/clip-vit-base-patch32"
MAX_LENGTH = 32

CASES = (
    "holding",
    "standing beside",
    "LEFT of right",
    "person's hand",
    "café touching 狗",
    "left/right — above!",
    ("very " * 80) + "near",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    slow = CLIPTokenizer.from_pretrained(TOKENIZER_ID)
    fast = CLIPTokenizerFast.from_pretrained(TOKENIZER_ID)

    slow_ids = slow(
        list(CASES),
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]
    fast_ids = fast(
        list(CASES),
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    if slow_ids != fast_ids:
        raise RuntimeError(
            "CLIPTokenizer and CLIPTokenizerFast token IDs diverged"
        )

    saved = fast.save_pretrained(out)
    tokenizer_json = out / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise RuntimeError(
            f"CLIPTokenizerFast did not write tokenizer.json: {saved}"
        )

    lines = []
    for text, ids in zip(CASES, slow_ids):
        if len(ids) > MAX_LENGTH:
            raise RuntimeError("reference CLIP tokenizer exceeded max length")
        if ids[0] != 49406 or ids[-1] != 49407:
            raise RuntimeError("reference CLIP BOS/EOS contract drifted")
        lines.append(
            text.encode("utf-8").hex()
            + "\t"
            + ",".join(str(int(value)) for value in ids)
        )

    golden = out / "clip-tokenizer-golden.txt"
    golden.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "schema": "kfcore.clip-tokenizer-golden/1",
        "tokenizer_id": TOKENIZER_ID,
        "max_length": MAX_LENGTH,
        "case_count": len(CASES),
        "tokenizer_json_sha256": sha256(tokenizer_json),
        "golden_sha256": sha256(golden),
        "bos_token_id": 49406,
        "eos_token_id": 49407,
        "student_pad_token_id": 0,
    }
    (out / "contract.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
