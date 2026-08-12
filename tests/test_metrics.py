from turkish_asr_eval.datasets import (
    compute_error_rates,
    compute_whisper_error_rates,
    clean_transcript_artifacts,
    normalize_text,
    normalize_text_whisper,
)


def test_compute_error_rates_returns_percentages():
    wer, cer = compute_error_rates("merhaba dunya", "merhaba")

    assert wer == 50.0
    assert cer > 0


def test_normalize_text_handles_turkish_case_punctuation_and_whitespace():
    assert normalize_text("  İSTANBUL, IĞDIR!\n") == "istanbul ığdır"


def test_compute_error_rates_ignores_formatting_differences():
    assert compute_error_rates("İstanbul, güzel!", "istanbul güzel") == (0.0, 0.0)


def test_clean_transcript_artifacts_removes_tokens_and_decomposed_dot():
    assert clean_transcript_artifacts("i̇stanbul <tr-TR>") == "istanbul  "
    assert compute_error_rates("istanbul", "i̇stanbul <tr-TR>") == (0.0, 0.0)


def test_whisper_normalization_is_available_as_a_separate_metric():
    assert normalize_text_whisper("Merhaba, Dünya!") == "merhaba dünya"
    assert compute_whisper_error_rates("Merhaba, Dünya!", "merhaba dünya") == (
        0.0,
        0.0,
    )
