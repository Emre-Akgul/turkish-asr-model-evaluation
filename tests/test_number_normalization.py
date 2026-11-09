import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from turkish_asr_eval import cli, number_normalization
from turkish_asr_eval.number_normalization import (
    compute_tn_error_rates,
    normalize_text_tn,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import recalculate_results  # noqa: E402

# A stand-in for the grammar: spells out the handful of forms the tests use, and
# reads upper case "ABD" letter by letter as the real grammar does.
FAKE_READINGS = {
    "20'si": "yirmisi",
    "6.5": "altı nokta beş",
    "2005": "iki bin beş",
    "ABD": "a be de",
}


def fake_grammar(text):
    fake_grammar.calls.append(text)
    if "boom" in text:
        raise ValueError("grammar failure")
    return " ".join(FAKE_READINGS.get(word, word) for word in text.split())


@pytest.fixture(autouse=True)
def fake_tn(monkeypatch):
    fake_grammar.calls = []
    monkeypatch.setattr(number_normalization, "_normalize_spoken", fake_grammar)
    monkeypatch.setattr(number_normalization, "_grammar_hash", "fakehash")
    number_normalization._spoken.cache_clear()
    yield
    number_normalization._spoken.cache_clear()


def test_digits_and_words_score_the_same():
    assert compute_tn_error_rates("yüzde 20'si geldi", "Yüzde yirmisi geldi.") == (
        0.0,
        0.0,
    )


def test_text_is_lowercased_before_the_grammar():
    # Upper case "ABD" would be spelled out by the grammar while the lower case
    # FLEURS reference "abd" would not; lowercasing first keeps both sides equal.
    assert normalize_text_tn("ABD'de") == normalize_text_tn("abd'de") == "abdde"
    assert fake_grammar.calls == ["abd'de"]


def test_artifacts_are_cleaned_before_the_grammar():
    assert normalize_text_tn("i̇stanbul <tr-TR>") == "istanbul"


def test_grammar_failure_falls_back_to_the_plain_text():
    assert normalize_text_tn("Boom, 6.5") == "boom 65"


def test_grammar_output_is_cached():
    normalize_text_tn("2005 yılında")
    normalize_text_tn("2005 yılında")
    assert fake_grammar.calls == ["2005 yılında"]


def test_cli_adds_tn_metrics_only_when_requested():
    spec = Namespace(
        name="fleurs",
        hf_name="google/fleurs",
        config="tr_tr",
        split="test",
        audio_column="audio",
        reference_column="transcription",
    )
    engine = Namespace(transcribe=lambda audio: "Sarsıntı 6.5 oldu")
    row = {"audio": b"", "transcription": "sarsıntı altı nokta beş oldu"}
    kwargs = dict(
        engine=engine,
        engine_name="fake",
        model="fake",
        spec=spec,
        reference_column="transcription",
    )

    plain = cli.evaluate_row(0, row, **kwargs)
    with_tn = cli.evaluate_row(0, row, tn=True, **kwargs)

    assert "tn_wer" not in plain
    assert with_tn["wer"] > 0
    assert (with_tn["tn_wer"], with_tn["tn_cer"]) == (0.0, 0.0)

    summary = cli.initial_summary()
    cli.update_summary(summary, with_tn)
    output = cli.summary_output(summary)
    assert output["mean_tn_wer"] == 0.0
    assert output["mean_wer"] > 0
    assert output["tn_grammar_hash"] == "fakehash"

    plain_summary = cli.initial_summary()
    cli.update_summary(plain_summary, plain)
    assert "mean_tn_wer" not in cli.summary_output(plain_summary)


def write_results(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


RECORDS = [
    {"reference": "kongre 2005 yılında", "prediction": "Kongre iki bin beş yılında."},
    {"reference": "sarsıntı oldu", "prediction": "", "error": "RuntimeError: x"},
]


def test_recalculate_adds_tn_metrics(tmp_path):
    path = tmp_path / "run.jsonl"
    write_results(path, RECORDS)

    output = recalculate_results.recalculate(path, tn=True)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["tn_wer"] == 0.0 and rows[0]["wer"] > 0
    assert "tn_wer" not in rows[1]
    assert output["rows"] == 2 and output["error_rows"] == 1
    assert output["mean_tn_wer"] == 0.0
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary == output


def test_recalculate_refuses_to_leave_tn_metrics_stale(tmp_path):
    path = tmp_path / "run.jsonl"
    write_results(path, RECORDS)
    recalculate_results.recalculate(path, tn=True)

    with pytest.raises(recalculate_results.StaleTNMetrics, match="--tn"):
        recalculate_results.recalculate(path)

    output = recalculate_results.recalculate(path, drop_tn=True)
    assert "mean_tn_wer" not in output
    assert "tn_wer" not in path.read_text()


def test_recalculate_without_tn_matches_the_cli_summary(tmp_path):
    path = tmp_path / "run.jsonl"
    write_results(path, RECORDS)
    output = recalculate_results.recalculate(path)

    summary = cli.initial_summary()
    for record in (json.loads(line) for line in path.read_text().splitlines()):
        cli.update_summary(summary, record)
    assert output == cli.summary_output(summary)
    assert not fake_grammar.calls


@pytest.mark.skipif(
    importlib.util.find_spec("nemo_text_processing") is None,
    reason="NeMo text processing is not installed",
)
def test_real_grammar(monkeypatch, tmp_path):
    monkeypatch.setenv(number_normalization.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(number_normalization, "_normalize_spoken", None)
    monkeypatch.setattr(number_normalization, "_grammar_hash", None)

    assert compute_tn_error_rates("yüzde 20'si", "yüzde yirmisi") == (0.0, 0.0)
    assert compute_tn_error_rates("4'ü", "dördü") == (0.0, 0.0)
    assert compute_tn_error_rates("sarsıntı 6.5", "sarsıntı altı nokta beş") == (
        0.0,
        0.0,
    )
    assert compute_tn_error_rates("ABD'de", "abd'de") == (0.0, 0.0)
