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
        "This is a short sentence.",
        "This is a much longer sentence designed to include multiple clauses, commas, and punctuation marks.",
        "What's happening?",
        "I'm testing contractions.",
        "Don't stop believing.",
        "We'll be there at 5:30 PM.",
        "The price is $19.99.",
        "Version 2.1.0 was released on 2026-02-20.",
        "Numbers: 0 1 2 3 10 100 9999.",
        "Coordinates: 37.7749, -122.4194.",
        "Email-like text: test@example.com",
        "URL-like text: https://example.com/path?q=test",
        "C'est la vie.",
        "naïve façade coöperate",
        "¿Cómo estás?",
        "Привет мир.",
        "こんにちは世界。",
        "你好，世界。",
        "مرحبا بالعالم.",
        "Tabs\tand\tspaces",
        "Multiple   spaces    here",
        "Leading-space-case",
        "Trailing space ",
        "Surrounding spaces",
        "New\nline",
        "Windows\r\nline ending",
        "Punctuation... wow!!!",
        "(parentheses) [brackets] {braces}",
        "Quotes: 'single' and \"double\"",
        "Slash / backslash \\\\ pipe |",
        "Math: 1+1=2, 3*7=21, 10/4=2.5",
        "Emoji 🙂🚀🎉",
        "Mixed emoji/text: Hello🙂world",
        "A",
        "I",
        "to",
        "on",
        "of",
        "The quick brown fox jumps over the lazy dog.",
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e., he paid a lot.",
        "U.S.A. vs UK vs EU",
        "10,000 users signed up in 24 hours.",
        "Model input IDs should be deterministic.",
        "Edge-case---with---dashes",
        "under_score and camelCaseTogether",
        "#hashtag @mention",
        "JSON-ish: {\"k\": [1,2,3], \"ok\": true}",
        "Code-ish: for(i=0;i<10;i++){sum+=i;}",
        "Sentence with ellipsis… and unicode punctuation—dash.",
        "Repeat repeat repeat repeat repeat.",
        "A very very very very very very very long sentence that keeps going to test segmentation behavior under longer contexts and ensure no pathological unknown-token loops appear in output ids.",
    ]

    generated = [
        f"Sample #{i}: value={i * i}, pct={i/100:.2f}, tag=T{i % 7}."
        for i in range(1, 31)
    ]
    return samples + generated


def check_parity(sp: spm.SentencePieceProcessor, hf: Tokenizer, samples: list[str]) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    for sample in samples:
        sp_ids = sp.encode(sample, out_type=int)
        hf_default_ids = hf.encode(sample).ids
        hf_no_special_ids = hf.encode(sample, add_special_tokens=False).ids

        if sp_ids != hf_default_ids or sp_ids != hf_no_special_ids:
            mismatches.append(
                Mismatch(
                    sample=sample,
                    sp_ids=sp_ids,
                    hf_default_ids=hf_default_ids,
                    hf_no_special_ids=hf_no_special_ids,
                )
            )
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
    parser = argparse.ArgumentParser(
        description="Validate parity between SentencePiece tokenizer.model and exported HF tokenizer.json"
    )
    parser.add_argument(
        "--spm-model",
        default="weights/tokenizer.model",
        help="Path to SentencePiece tokenizer.model",
    )
    parser.add_argument(
        "--tokenizer-json",
        default="hf/tokenizer.json",
        help="Path to exported tokenizer.json",
    )
    parser.add_argument(
        "--determinism-runs",
        type=int,
        default=5,
        help="How many repeated encoding runs per sample",
    )
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
        print(f"❌ Parity failed on {len(mismatches)} samples")
        for mismatch in mismatches[:10]:
            print("---")
            print(f"Sample: {mismatch.sample!r}")
            print(f"SPM: {mismatch.sp_ids}")
            print(f"HF default: {mismatch.hf_default_ids}")
            print(f"HF no-special: {mismatch.hf_no_special_ids}")
    else:
        print("✅ ID parity passed for all samples")

    if deterministic:
        print(f"✅ Deterministic across {args.determinism_runs} runs")
    else:
        print(f"❌ Non-deterministic output detected for sample: {sample!r}")

    return 0 if not mismatches and deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
