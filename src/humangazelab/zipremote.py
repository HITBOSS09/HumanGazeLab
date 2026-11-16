"""Read selected members from a local or remote ZIP without downloading the archive.

HTTP sources use range requests. The central directory is read from the end of
the file. Extraction uses the central-directory sizes and checks CRC-32.
"""

from __future__ import annotations

import os
import struct
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZipEntry:
    filename: str
    method: int
    crc: int
    comp_size: int
    uncomp_size: int
    local_offset: int


class RangeReader:
    def __init__(self, source: str, timeout: float = 120.0, retries: int = 4):
        self.source = source
        self.timeout = timeout
        self.retries = retries
        self._length: int | None = None

    def content_length(self) -> int:
        if self._length is None:
            if self.source.startswith("http://") or self.source.startswith("https://"):
                self._length = _http_length(self.source, self.timeout)
            else:
                self._length = os.path.getsize(self.source)
        return self._length

    def __call__(self, start: int, end: int) -> bytes:
        if end < start:
            return b""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                if self.source.startswith("http://") or self.source.startswith("https://"):
                    return _http_range(self.source, start, end, self.timeout)
                with open(self.source, "rb") as handle:
                    handle.seek(start)
                    return handle.read(end - start + 1)
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 == self.retries:
                    break
        raise RuntimeError(f"failed to read {self.source} bytes {start}-{end}: {last_error}") from last_error


def parse_central_directory(reader: RangeReader) -> list[ZipEntry]:
    length = reader.content_length()
    window = min(length, 65535 + 22)
    tail = reader(length - window, length - 1)
    signature = tail.rfind(b"PK\x05\x06")
    if signature < 0:
        raise ValueError("ZIP end-of-central-directory record was not found in the last 64 KiB")
    eocd = tail[signature:]
    _disk_no, _disk_cd, _disk_entries, total_entries, cd_size, cd_offset, _comment_len = struct.unpack_from(
        "<HHHHIIH", eocd, 4
    )
    if total_entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 central directories are not implemented")
    blob = reader(cd_offset, cd_offset + cd_size - 1)
    entries: list[ZipEntry] = []
    pos = 0
    while pos + 46 <= len(blob) and blob[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", blob, pos + 10)[0]
        crc = struct.unpack_from("<I", blob, pos + 16)[0]
        comp_size = struct.unpack_from("<I", blob, pos + 20)[0]
        uncomp_size = struct.unpack_from("<I", blob, pos + 24)[0]
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", blob, pos + 28)
        local_offset = struct.unpack_from("<I", blob, pos + 42)[0]
        filename = blob[pos + 46 : pos + 46 + name_len].decode("utf-8")
        entries.append(
            ZipEntry(
                filename=filename,
                method=method,
                crc=crc,
                comp_size=comp_size,
                uncomp_size=uncomp_size,
                local_offset=local_offset,
            )
        )
        pos += 46 + name_len + extra_len + comment_len
    if len(entries) != total_entries:
        raise ValueError(f"central directory listed {total_entries} entries but parsed {len(entries)}")
    return entries


def extract_entry(reader: RangeReader, entry: ZipEntry) -> bytes:
    header = reader(entry.local_offset, entry.local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"{entry.filename} is missing a local file header")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    data_start = entry.local_offset + 30 + name_len + extra_len
    compressed = reader(data_start, data_start + entry.comp_size - 1) if entry.comp_size else b""
    if entry.method == 0:
        raw = compressed
    elif entry.method == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"{entry.filename} uses unsupported compression method {entry.method}")
    if len(raw) != entry.uncomp_size:
        raise ValueError(f"{entry.filename} uncompressed size is {len(raw)}, expected {entry.uncomp_size}")
    if (zlib.crc32(raw) & 0xFFFFFFFF) != entry.crc:
        raise ValueError(f"{entry.filename} CRC-32 does not match the central directory")
    return raw


def jpeg_index(entries: list[ZipEntry]) -> dict[str, list[ZipEntry]]:
    """Map JPEG basename to archive members, skipping macOS metadata."""
    index: dict[str, list[ZipEntry]] = {}
    for entry in entries:
        if not entry.filename.lower().endswith(".jpg"):
            continue
        if entry.filename.startswith("__MACOSX") or "/._" in entry.filename:
            continue
        base = entry.filename.rsplit("/", 1)[-1]
        index.setdefault(base, []).append(entry)
    return index


def _http_length(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "HumanGazeLab/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


def _http_range(url: str, start: int, end: int, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "HumanGazeLab/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise OSError(f"expected HTTP 206 for a range request, got {response.status}")
        return response.read()
