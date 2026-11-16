import zipfile
from pathlib import Path

from humangazelab.zipremote import RangeReader, extract_entry, parse_central_directory


def test_range_reader_roundtrip_on_a_local_zip(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    payload = b"fixation-bytes-0123456789"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("images/knife/000000000194.jpg", payload)
        handle.writestr("images/readme.txt", b"not-an-image")
    reader = RangeReader(str(archive))
    entries = parse_central_directory(reader)
    jpegs = [entry for entry in entries if entry.filename.endswith(".jpg")]
    assert len(jpegs) == 1
    assert extract_entry(reader, jpegs[0]) == payload
