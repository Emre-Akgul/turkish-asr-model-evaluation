import json
from argparse import Namespace

import pytest

from turkish_asr_eval import cli


class FakeDataset:
    column_names = ["audio", "transcription"]

    def __iter__(self):
        yield {"audio": "sample.wav", "transcription": "merhaba"}

    def __len__(self):
        return 1


class FakeEngine:
    def __init__(self, model):
        self.model = model

    def load(self):
        self.loaded = True

    def transcribe(self, audio):
        return "merhaba"


def test_cli_parser_accepts_required_args():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--engine",
            "faster_whisper",
            "--model",
            "small",
            "--dataset",
            "fleurs:test",
        ]
    )

    assert args.engine == "faster_whisper"
    assert args.model == "small"
    assert args.dataset == "fleurs:test"
    assert args.device is None
    assert args.compute_type is None
    assert args.workers == 1
    assert args.limit is None


def test_bad_engine_fails_before_model_loading():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--engine",
                "bad",
                "--model",
                "model",
                "--dataset",
                "fleurs:test",
            ]
        )


def test_run_writes_jsonl_with_mocked_engine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "load_hf_dataset", lambda spec, streaming=True: FakeDataset()
    )
    monkeypatch.setattr(
        cli, "create_engine", lambda engine, model, **options: FakeEngine(model)
    )

    args = Namespace(
        engine="faster_whisper",
        model="fake-model",
        dataset="fleurs:validation",
        device=None,
        compute_type=None,
        name="test-run",
        workers=1,
        limit=None,
    )

    output_path = cli.run(args)

    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record["prediction"] == "merhaba"
    assert record["reference"] == "merhaba"
    assert record["wer"] == 0
    assert record["cer"] == 0
    assert record["whisper_wer"] == 0
    assert record["whisper_cer"] == 0
    assert (tmp_path / "results" / "test-run.summary.json").exists()


def test_run_limits_rows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class MultiRowDataset:
        column_names = ["audio", "transcription"]

        def __iter__(self):
            for index in range(3):
                yield {"audio": f"sample-{index}.wav", "transcription": "merhaba"}

        def __len__(self):
            return 3

    monkeypatch.setattr(
        cli, "load_hf_dataset", lambda spec, streaming=True: MultiRowDataset()
    )
    monkeypatch.setattr(
        cli, "create_engine", lambda engine, model, **options: FakeEngine(model)
    )

    args = Namespace(
        engine="faster_whisper",
        model="fake-model",
        dataset="fleurs:validation",
        device=None,
        compute_type=None,
        name="limited-run",
        workers=1,
        limit=2,
    )

    output_path = cli.run(args)

    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2


def test_summary_uses_corpus_rates_instead_of_sentence_average(tmp_path):
    summary = cli.initial_summary()
    cli.update_summary(
        summary,
        {"reference": "bir", "prediction": "yanlış", "wer": 100.0, "cer": 100.0},
    )
    cli.update_summary(
        summary,
        {
            "reference": "iki üç dört beş altı yedi sekiz dokuz on",
            "prediction": "iki üç dört beş altı yedi sekiz dokuz on",
            "wer": 0.0,
            "cer": 0.0,
        },
    )
    output_path = tmp_path / "summary.json"
    cli.write_summary(output_path, summary)
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert output["mean_wer"] == 10.0
    assert "mean_whisper_wer" in output
    assert "mean_whisper_cer" in output
