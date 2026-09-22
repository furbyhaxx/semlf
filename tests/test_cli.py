import sys

import pytest

from semif_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--backend", "mlx", "--mode", "reranker"], "reranker requires torch"),
    (["--mode", "direct", "--mlx-bits", "4"], "requires --backend mlx"),
    (["--mode", "direct", "--mlx-cache-limit-mib", "0"], "requires --backend mlx"),
    (["--mode", "direct", "--backend", "mlx", "--mlx-cache-limit-mib", "-1"], "must be nonnegative"),
])
def test_invalid_backend_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["semif-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize('limit', [None, 0, 512])
def test_cli_passes_cache_limit_to_loader(tmp_path, monkeypatch, limit):
    import json
    from types import SimpleNamespace
    import semif_phase1

    fake_backend = SimpleNamespace(
        DEFAULT_CACHE_LIMIT_MIB=256,
        load_model=lambda source, revision, bits, *, cache_limit_mib:
            (None, None, {'limit': cache_limit_mib}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(semif_phase1, 'mlx_backend', fake_backend, raising=False)
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    source.write_text(json.dumps({'id': 'test', 'state': 'Evidence', 'question': 'Supported?',
                                 'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}) + '\n')
    args = ['semif-score', '--backend', 'mlx', '--mode', 'direct', '--model', 'unused',
            '--revision', 'unused', '--input', str(source), '--output', str(output)]
    if limit is not None:
        args += ['--mlx-cache-limit-mib', str(limit)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert json.loads(output.read_text())['limit'] == (256 if limit is None else limit)


def test_mlx_rejects_image_rows_before_loading(tmp_path, monkeypatch, capsys):
    import json

    png = tmp_path / "shot.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({
        "id": "test",
        "state": "Evidence",
        "question": "Supported?",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        "Image": str(png),
    }) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "mlx", "--mode", "direct", "--model", "unused",
        "--revision", "unused", "--input", str(source), "--output", str(output),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Image" in capsys.readouterr().err
    assert not output.exists()
