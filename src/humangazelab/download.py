"""Download the public COCO-FreeView fixation release and record what came back."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from humangazelab.config import AnalysisConfig, load_config

FALLBACK = """
Automated download failed. The official pages are:
  COCO-FreeView: https://sites.google.com/view/cocosearch/coco-freeview
  COCO-Search18 stimuli: https://sites.google.com/view/cocosearch/home

Place these files by hand and rerun the scripts:
  data/raw/COCOFreeView_fixations_trainval.json
      from http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json
  data/raw/COCOFreeView_readme.txt
      from the Google Drive readme linked on the COCO-FreeView page
  Optional local image archives, then:
      python scripts/prepare_images.py --tp-zip /path/to/COCOSearch18-images-TP.zip --ta-zip /path/to/COCOSearch18-images-TA.zip
"""


def download(root: Path, config: AnalysisConfig, force: bool = False) -> dict:
    root = Path(root)
    fixation_path = root / config.fixation_path
    readme_path = root / config.readme_path
    fixation_path.parent.mkdir(parents=True, exist_ok=True)
    output = root / config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fixation = _ensure_file(config.fixation_url, fixation_path, root, force=force, expect_json=True)
    readme = _ensure_file(config.readme_url, readme_path, root, force=force, expect_text="Data format")
    test_probe = _probe(config.test_probe_url)
    usage = shutil.disk_usage(root)
    committed_readme = root / "docs" / "sources" / "COCO-FreeView-readme.txt"
    committed_hash = _sha256(committed_readme) if committed_readme.exists() else ""
    record = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_page": config.project_page,
        "stimulus_page": config.stimulus_page,
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
        "fixation": fixation,
        "readme": readme,
        "readme_matches_committed_copy": bool(committed_hash) and committed_hash == readme["sha256"],
        "committed_readme_sha256": committed_hash,
        "test_fixation_probe": test_probe,
        "image_archives": {
            "tp_zip_url": config.tp_zip_url,
            "ta_zip_url": config.ta_zip_url,
            "note": "Image bytes are fetched by range request in scripts/prepare_images.py. The full zip is not stored.",
        },
    }
    (output / "download_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _ensure_file(url: str, dest: Path, root: Path, force: bool, expect_json: bool = False, expect_text: str = "") -> dict:
    headers = _head(url)
    remote_length = _content_length(headers)
    if dest.exists() and not force and (remote_length is None or dest.stat().st_size == remote_length):
        payload = dest.read_bytes()
        _validate_payload(payload, dest, expect_json, expect_text)
        return _file_record(url, dest, root, headers, downloaded=False)
    try:
        payload, status, response_headers = _get(url)
    except Exception as exc:
        raise RuntimeError(f"could not download {url}\n{FALLBACK}") from exc
    _validate_payload(payload, dest, expect_json, expect_text)
    dest.write_bytes(payload)
    headers = {**headers, **response_headers, "status": status}
    return _file_record(url, dest, root, headers, downloaded=True)


def _validate_payload(payload: bytes, dest: Path, expect_json: bool, expect_text: str) -> None:
    stripped = payload.lstrip()
    if stripped.startswith(b"<!DOCTYPE") or stripped.lower().startswith(b"<html"):
        raise RuntimeError(f"{dest.name} download returned HTML rather than the dataset file.\n{FALLBACK}")
    if expect_json and not stripped.startswith(b"["):
        raise RuntimeError(f"{dest.name} does not start with a JSON list.\n{FALLBACK}")
    if expect_text and expect_text.encode() not in payload:
        raise RuntimeError(f"{dest.name} does not contain the expected readme marker {expect_text!r}.\n{FALLBACK}")


def _file_record(url: str, dest: Path, root: Path, headers: dict, downloaded: bool) -> dict:
    try:
        rel = dest.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = dest.name
    return {
        "url": url,
        "path": rel,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "downloaded_this_run": downloaded,
        "content_length": _content_length(headers),
        "last_modified": headers.get("Last-Modified") or headers.get("last-modified"),
        "etag": headers.get("ETag") or headers.get("etag"),
        "content_type": headers.get("Content-Type") or headers.get("content-type"),
    }


def _head(url: str) -> dict:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "HumanGazeLab/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return {key: value for key, value in response.headers.items()}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"head_error": str(exc)}


def _get(url: str) -> tuple[bytes, int, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "HumanGazeLab/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read(), int(response.status), {key: value for key, value in response.headers.items()}


def _probe(url: str) -> dict:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "HumanGazeLab/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return {"url": url, "http_status": int(response.status)}
    except urllib.error.HTTPError as exc:
        return {
            "url": url,
            "http_status": int(exc.code),
            "note": "The COCO-FreeView project page says the testing set was taken down pending an online evaluation service.",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"url": url, "http_status": None, "error": str(exc)}


def _content_length(headers: dict):
    raw = headers.get("Content-Length") or headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download the public COCO-FreeView fixation file and readme.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    record = download(Path(args.root), load_config(args.config), force=args.force)
    print(json.dumps({
        "fixation_sha256": record["fixation"]["sha256"],
        "fixation_bytes": record["fixation"]["bytes"],
        "readme_sha256": record["readme"]["sha256"],
        "readme_matches_committed_copy": record["readme_matches_committed_copy"],
        "test_probe": record["test_fixation_probe"],
        "disk_free_bytes": record["disk_free_bytes"],
    }, indent=2))
