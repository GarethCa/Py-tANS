import pytest

from pytans.cli import main


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "data.txt"
    path.write_bytes(b"how much wood would a woodchuck chuck " * 100)
    return path


def test_compress_then_decompress(sample_file, tmp_path):
    original = sample_file.read_bytes()

    assert main(["compress", str(sample_file)]) == 0
    compressed = sample_file.with_suffix(".txt.tans")
    assert compressed.exists()
    assert compressed.stat().st_size < len(original)

    sample_file.unlink()
    assert main(["decompress", str(compressed)]) == 0
    assert sample_file.read_bytes() == original


def test_explicit_output_and_table_log(sample_file, tmp_path):
    out = tmp_path / "out.bin"
    assert main(["compress", str(sample_file), "-o", str(out), "--table-log", "8"]) == 0
    restored = tmp_path / "restored.txt"
    assert main(["decompress", str(out), "-o", str(restored)]) == 0
    assert restored.read_bytes() == sample_file.read_bytes()


def test_refuses_overwrite_without_force(sample_file, tmp_path):
    out = tmp_path / "out.bin"
    out.write_bytes(b"precious")
    with pytest.raises(SystemExit):
        main(["compress", str(sample_file), "-o", str(out)])
    assert out.read_bytes() == b"precious"
    assert main(["compress", str(sample_file), "-o", str(out), "--force"]) == 0


def test_decompress_needs_derivable_name(tmp_path):
    bogus = tmp_path / "file.bin"
    bogus.write_bytes(b"\x00")
    with pytest.raises(SystemExit):
        main(["decompress", str(bogus)])


def test_decompress_rejects_garbage(tmp_path):
    bogus = tmp_path / "file.tans"
    bogus.write_bytes(b"this is not a tans frame")
    with pytest.raises(SystemExit):
        main(["decompress", str(bogus)])


def test_stdin_stdout(sample_file, capsysbinary, monkeypatch):
    import io
    import sys

    data = sample_file.read_bytes()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    assert main(["compress", "-"]) == 0
    blob = capsysbinary.readouterr().out

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(blob)))
    assert main(["decompress", "-"]) == 0
    assert capsysbinary.readouterr().out == data
