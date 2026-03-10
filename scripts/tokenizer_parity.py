import argparse
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer


@dataclass
class Mismatch:
    sample: str
    sp_ids: list[int]
    hf_default_ids: list[int]
    hf_no_special_ids: list[int]


def build_samples() -> list[str]:
    samples = [
        "Hello world.",
        "hello world.",
        "HELLO WORLD!",
        "Pocket TTS is streaming.",
        "What's happening?",
        "I'm testing contractions.",
        "Version 2.1.0 was released on 2026-02-20.",
        "Coordinates: 37.7749, -122.4194.",
        "URL-like text: https://example.com/path?q=test",
        "C'est la vie.",
        "naive facade cooperate",
        "Tabs\tand\tspaces",
        "New\nline",
        "Windows\r\nline ending",
        "JSON-ish: {\"k\": [1,2,3], \"ok\": true}",
        "Code-ish: for(i=0;i<10;i++){sum+=i;}",
        "The quick brown fox jumps over the lazy dog.",
        "Model input IDs should be deterministic.",
        "Edge-case---with---dashes",
    ]
    generated = [f"Sample #{index}: value={index * index}, pct={index / 100:.2f}, tag=T{index % 7}." for index in range(1, 31)]
    return samples + generated


def check_parity(sp: spm.SentencePieceProcessor, hf: Tokenizer, samples: list[str]) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    for sample in samples:
        sp_ids = sp.encode(sample, out_type=int)
        hf_default_ids = hf.encode(sample).ids
        hf_no_special_ids = hf.encode(sample, add_special_tokens=False).ids
        if sp_ids != hf_default_ids or sp_ids != hf_no_special_ids:
            mismatches.append(Mismatch(sample, sp_ids, hf_default_ids, hf_no_special_ids))
    return mismatches


def check_determinism(hf: Tokenizer, samples: list[str], runs: int) -> tuple[bool, str | None]:
    for sample in samples:
        first_default = hf.encode(sample).ids
        first_no_special = hf.encode(sample, add_special_tokens=False).ids
        for _ in range(runs - 1):
            if hf.encode(sample).ids != first_default:
                return False, sample
            if hf.encode(sample, add_special_tokens=False).ids != first_no_special:
                return False, sample
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate parity between SentencePiece tokenizer.model and exported HF tokenizer.json")
    parser.add_argument("--spm-model", default="weights/tokenizer.model", help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--tokenizer-json", default="hf/tokenizer.json", help="Path to exported tokenizer.json")
    parser.add_argument("--determinism-runs", type=int, default=5, help="How many repeated encoding runs per sample")
    args = parser.parse_args()

    spm_path = Path(args.spm_model)
    tokenizer_path = Path(args.tokenizer_json)
    if not spm_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    hf = Tokenizer.from_file(str(tokenizer_path))
    samples = build_samples()

    mismatches = check_parity(sp, hf, samples)
    deterministic, sample = check_determinism(hf, samples, args.determinism_runs)

    print(f"Checked {len(samples)} samples")
    if mismatches:
        print(f"Parity failed on {len(mismatches)} samples")
        for mismatch in mismatches[:10]:
            print("---")
            print(f"Sample: {mismatch.sample!r}")
            print(f"SPM: {mismatch.sp_ids}")
            print(f"HF default: {mismatch.hf_default_ids}")
            print(f"HF no-special: {mismatch.hf_no_special_ids}")
    else:
        print("ID parity passed for all samples")

    if deterministic:
        print(f"Deterministic across {args.determinism_runs} runs")
    else:
        print(f"Non-deterministic output detected for sample: {sample!r}")

    return 0 if not mismatches and deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())