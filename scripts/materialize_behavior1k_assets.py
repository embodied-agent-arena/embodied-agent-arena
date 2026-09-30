#!/usr/bin/env python3
"""Download and verify official BEHAVIOR-1K assets without accepting licenses implicitly."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile
import zlib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _configured_root(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    if not value:
        return default
    configured = Path(value).expanduser()
    return (configured if configured.is_absolute() else REPOSITORY_ROOT / configured).resolve()


EXTERNAL_ROOT = _configured_root("EMBODIED_ARENA_EXTERNAL_ROOT", REPOSITORY_ROOT / "external")
ARTIFACT_ROOT = _configured_root("EMBODIED_ARENA_ARTIFACT_ROOT", REPOSITORY_ROOT / "artifacts")
DEFAULT_ASSET_ROOT = EXTERNAL_ROOT / "assets" / "behavior1k" / "datasets"
DEFAULT_CACHE_ROOT = ARTIFACT_ROOT / "runtime-cache" / "behavior1k-assets-hf"
HF_REPOSITORY = "behavior-1k/zipped-datasets"
HF_REPOSITORY_TYPE = "dataset"

# The restricted archive is intentionally not downloaded in full for a
# selected-case closure.  Pin its byte length and ZIP central directory so a
# mutable remote ``main`` cannot silently change what the range reader sees.
BEHAVIOR_ARCHIVE_SIZE = 31_457_673_073
BEHAVIOR_CENTRAL_DIRECTORY_OFFSET = 31_437_414_975
BEHAVIOR_CENTRAL_DIRECTORY_SIZE = 20_258_000
BEHAVIOR_CENTRAL_DIRECTORY_SHA256 = (
    "f27325eee54ec5286171c2599f6e2bfb9028feb5f39ddf72ce48ff8d6ca10a65"
)

STRUCTURE_CATEGORIES = frozenset(
    {"background", "ceilings", "driveway", "fence", "floors", "lawn", "roof", "walls"}
)
GROUND_CATEGORIES = frozenset({"carpet", "driveway", "floors", "lawn"})
ALWAYS_LOADED_CATEGORIES = frozenset({"door", "sliding_door"})
SELECTIVE_CASES = {
    "turning_on_radio": {
        "scene_model": "house_double_floor_lower",
        "template": "house_double_floor_lower_task_turning_on_radio_0_0_template.json",
        # Keep this aligned with the benchmark's lightweight live config.
        "not_load_object_categories": frozenset({"ceilings"}),
    }
}

COMPONENTS = {
    "robot": {
        "filename": "omnigibson-robot-assets.zip",
        "sha256": "5e9fe726172c8b4beb4ba4e12dbf5cf6c5791553a05c52be661eb2d6a3d241b8",
        "target": "omnigibson-robot-assets",
        "license_key_required": False,
    },
    "challenge": {
        "filename": "2025-challenge-task-instances.zip",
        "sha256": "6262a496c82a534f42166ca1569ef28dc4cd9e53ffd0f9f8b0e7ecb8c62acde3",
        "target": "2025-challenge-task-instances",
        "license_key_required": False,
    },
    "behavior": {
        "filename": "behavior-1k-assets-3.9.0.zip",
        "sha256": "09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e",
        "target": "behavior-1k-assets",
        "license_key_required": True,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_key(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("--key-file must be an existing regular file, not a symlink")
    key = path.read_bytes().strip()
    if len(key) != 44 or not key.endswith(b"="):
        raise RuntimeError("--key-file is not a valid 44-byte Fernet key")
    try:
        import base64

        decoded = base64.urlsafe_b64decode(key)
    except Exception as exc:
        raise RuntimeError("--key-file is not valid URL-safe base64") from exc
    if len(decoded) != 32:
        raise RuntimeError("--key-file does not decode to a 32-byte Fernet key")
    return key


def _download(component: str, cache_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    spec = COMPONENTS[component]
    cache_root.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=HF_REPOSITORY,
            filename=str(spec["filename"]),
            repo_type=HF_REPOSITORY_TYPE,
            local_dir=cache_root,
        )
    )
    actual = _sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"official {component} archive digest mismatch: expected {spec['sha256']}, got {actual}"
        )
    return path


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if destination != target_root and target_root not in destination.parents:
                raise RuntimeError(f"archive member escapes target directory: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"archive contains unsupported symbolic link: {member.filename}")
        bundle.extractall(target)


class _HTTPRangeReader(io.RawIOBase):
    """Small seekable HTTP reader with one-block caching for ``zipfile``."""

    def __init__(self, url: str, *, expected_size: int, block_size: int = 8 * 1024 * 1024):
        super().__init__()
        self._original_url = url
        self._resolved_url = url
        self._block_size = max(64 * 1024, int(block_size))
        self._position = 0
        self._size = 0
        self._cache_start = 0
        self._cache = b""
        self.bytes_transferred = 0
        self.range_digests: dict[tuple[int, int], str] = {}

        payload, resolved_url, total_size = self._request(url, 0, 0)
        if len(payload) != 1:
            raise RuntimeError("official BEHAVIOR archive returned an invalid range probe")
        if total_size != expected_size:
            raise RuntimeError(
                "official BEHAVIOR archive size mismatch: "
                f"expected {expected_size}, got {total_size}"
            )
        self._resolved_url = resolved_url
        self._size = total_size

    @staticmethod
    def _request(url: str, start: int, end: int) -> tuple[bytes, str, int]:
        request = Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "embodied-arena-selective-assets/1",
            },
        )
        with urlopen(request, timeout=180) as response:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise RuntimeError(
                    f"official BEHAVIOR archive does not support byte ranges (HTTP {status})"
                )
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if match is None:
                raise RuntimeError("official BEHAVIOR archive returned an invalid Content-Range")
            actual_start, actual_end, total_size = (int(value) for value in match.groups())
            if (actual_start, actual_end) != (start, end):
                raise RuntimeError(
                    "official BEHAVIOR archive returned the wrong byte range: "
                    f"expected {start}-{end}, got {actual_start}-{actual_end}"
                )
            payload = response.read()
            if len(payload) != end - start + 1:
                raise RuntimeError(
                    "official BEHAVIOR archive returned a short byte range: "
                    f"expected {end - start + 1}, got {len(payload)}"
                )
            return payload, response.geturl(), total_size

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def _fetch(self, start: int, end: int) -> bytes:
        try:
            payload, resolved_url, total_size = self._request(self._resolved_url, start, end)
        except Exception:
            # Signed CDN URLs expire. Re-resolve through the stable Hub URL once.
            payload, resolved_url, total_size = self._request(self._original_url, start, end)
        if total_size != self._size:
            raise RuntimeError("official BEHAVIOR archive changed during selective extraction")
        self._resolved_url = resolved_url
        self.bytes_transferred += len(payload)
        self.range_digests[(start, len(payload))] = hashlib.sha256(payload).hexdigest()
        return payload

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._position
        size = min(size, self._size - self._position)
        if size <= 0:
            return b""

        start = self._position
        end = start + size
        cache_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= start and end <= cache_end):
            if size >= self._block_size:
                fetch_start, fetch_end = start, end - 1
            else:
                # Start at the requested position so a small read straddling an
                # aligned block boundary is still fully covered.
                fetch_start = start
                fetch_end = min(self._size, fetch_start + self._block_size) - 1
            self._cache_start = fetch_start
            self._cache = self._fetch(fetch_start, fetch_end)

        relative_start = start - self._cache_start
        payload = self._cache[relative_start : relative_start + size]
        if len(payload) != size:
            raise RuntimeError("internal HTTP range cache returned a short read")
        self._position += size
        return payload


def _case_template_path(asset_root: Path, case_name: str) -> Path:
    case = SELECTIVE_CASES[case_name]
    return (
        asset_root
        / "2025-challenge-task-instances"
        / "scenes"
        / str(case["scene_model"])
        / "json"
        / str(case["template"])
    )


def _case_member_prefixes(asset_root: Path, case_name: str) -> tuple[set[str], tuple[str, ...]]:
    case = SELECTIVE_CASES[case_name]
    template_path = _case_template_path(asset_root, case_name)
    if not template_path.is_file():
        raise RuntimeError(
            f"selected case requires the official challenge template: {template_path}"
        )
    try:
        template = json.loads(template_path.read_text(encoding="utf-8"))
        relevant_names = set(template["metadata"]["task"]["inst_to_name"].values())
        objects = template["objects_info"]["init_info"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid official challenge template: {template_path}") from exc
    if not isinstance(objects, dict):
        raise RuntimeError(f"invalid object closure in official challenge template: {template_path}")

    excluded = set(case["not_load_object_categories"])
    model_prefixes: set[str] = set()
    for object_name, object_info in objects.items():
        args = object_info.get("args", {}) if isinstance(object_info, dict) else {}
        category = args.get("category", "object")
        model = args.get("model")
        # Match InteractiveTraversableScene._should_load_object exactly.  Its
        # final ``or is_building_structure`` intentionally overrides the
        # category blacklist, so ceilings still need their encrypted USD at
        # scene prebuild time even though the lightweight case blacklists them.
        is_building_structure = (
            category in (STRUCTURE_CATEGORIES - GROUND_CATEGORIES)
            or category in ALWAYS_LOADED_CATEGORIES
        )
        task_or_structure = object_name in relevant_names or category in STRUCTURE_CATEGORIES
        should_load = (category not in excluded and task_or_structure) or is_building_structure
        if not should_load:
            continue
        if not isinstance(category, str) or not isinstance(model, str):
            raise RuntimeError(f"selected case has an invalid model entry: {object_name}")
        if re.fullmatch(r"[a-z0-9_]+", category) is None or re.fullmatch(r"[a-z0-9_]+", model) is None:
            raise RuntimeError(f"selected case has an unsafe model entry: {category}/{model}")
        model_prefixes.add(f"objects/{category}/{model}/")

    if not model_prefixes:
        raise RuntimeError(f"selected case resolved to an empty object closure: {case_name}")
    prefixes = (
        "metadata/",
        # OmniGibson enumerates this directory while constructing visual
        # semantic observations, even when the selected task does not spawn a
        # physical system.  Keep the official system definitions in the
        # selected-case closure instead of manufacturing an empty directory.
        "systems/",
        f"scenes/{case['scene_model']}/",
        *sorted(model_prefixes),
    )
    return {"VERSION"}, prefixes


def _validate_member_destination(target: Path, member: zipfile.ZipInfo) -> Path:
    target_root = target.resolve()
    destination = (target / member.filename).resolve()
    if destination != target_root and target_root not in destination.parents:
        raise RuntimeError(f"archive member escapes target directory: {member.filename}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeError(f"archive contains unsupported symbolic link: {member.filename}")
    if member.flag_bits & 1:
        raise RuntimeError(f"archive contains unexpected ZIP-level encryption: {member.filename}")
    return destination


def _extract_selected_member(
    bundle: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path
) -> None:
    destination = _validate_member_destination(target, member)
    if member.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        return
    if _existing_member_matches(destination, member):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
            with bundle.open(member, "r") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _existing_member_matches(destination: Path, member: zipfile.ZipInfo) -> bool:
    """Avoid re-fetching a previously verified range-extracted member."""

    if destination.is_symlink() or not destination.is_file():
        return False
    if destination.stat().st_size != member.file_size:
        return False
    checksum = 0
    with destination.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF == member.CRC


def materialize_selective_case(
    case_name: str,
    *,
    asset_root: Path,
    key_file: Path,
    block_size: int = 8 * 1024 * 1024,
) -> dict[str, object]:
    """Range-extract the exact asset closure used by one lightweight case."""

    key = _validated_key(key_file)
    exact_names, prefixes = _case_member_prefixes(asset_root, case_name)
    from huggingface_hub import hf_hub_url

    spec = COMPONENTS["behavior"]
    url = hf_hub_url(
        repo_id=HF_REPOSITORY,
        filename=str(spec["filename"]),
        repo_type=HF_REPOSITORY_TYPE,
    )
    reader = _HTTPRangeReader(
        url,
        expected_size=BEHAVIOR_ARCHIVE_SIZE,
        block_size=block_size,
    )
    target = asset_root / str(spec["target"])
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(reader) as bundle:
        members = [
            member
            for member in bundle.infolist()
            if member.filename in exact_names
            or any(member.filename.startswith(prefix) for prefix in prefixes)
        ]
        index_digest = reader.range_digests.get(
            (BEHAVIOR_CENTRAL_DIRECTORY_OFFSET, BEHAVIOR_CENTRAL_DIRECTORY_SIZE)
        )
        if index_digest != BEHAVIOR_CENTRAL_DIRECTORY_SHA256:
            raise RuntimeError(
                "official BEHAVIOR archive central-directory digest mismatch: "
                f"expected {BEHAVIOR_CENTRAL_DIRECTORY_SHA256}, got {index_digest}"
            )
        selected_names = {member.filename for member in members}
        missing_prefixes = [
            prefix for prefix in prefixes if not any(name.startswith(prefix) for name in selected_names)
        ]
        if exact_names - selected_names or missing_prefixes:
            raise RuntimeError(
                "official BEHAVIOR archive is missing selected-case members: "
                f"exact={sorted(exact_names - selected_names)}, prefixes={missing_prefixes}"
            )
        for member in members:
            _extract_selected_member(bundle, member, target)

    key_target = asset_root / "omnigibson.key"
    key_target.write_bytes(key + b"\n")
    key_target.chmod(0o600)
    template_path = _case_template_path(asset_root, case_name)
    manifest = {
        "schema_version": 1,
        "case": case_name,
        "source": {
            "repository": HF_REPOSITORY,
            "filename": spec["filename"],
            "archive_sha256": spec["sha256"],
            "archive_bytes": BEHAVIOR_ARCHIVE_SIZE,
            "central_directory_sha256": BEHAVIOR_CENTRAL_DIRECTORY_SHA256,
        },
        "challenge_template": {
            "path": str(template_path.relative_to(asset_root)),
            "sha256": _sha256_file(template_path),
        },
        "closure": {
            "members": len(members),
            "compressed_bytes": sum(member.compress_size for member in members),
            "expanded_bytes": sum(member.file_size for member in members),
            "http_bytes_transferred": reader.bytes_transferred,
        },
    }
    manifest_path = target / f".selected-case-{case_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def materialize(
    components: list[str],
    *,
    asset_root: Path,
    cache_root: Path,
    key_file: Path | None,
) -> None:
    requires_key = any(bool(COMPONENTS[name]["license_key_required"]) for name in components)
    key = None
    if requires_key:
        if key_file is None:
            raise RuntimeError(
                "the official BEHAVIOR data bundle requires --key-file; this tool never accepts its license "
                "or downloads the restricted key on the user's behalf"
            )
        key = _validated_key(key_file)

    asset_root.mkdir(parents=True, exist_ok=True)
    for component in components:
        spec = COMPONENTS[component]
        archive = _download(component, cache_root)
        _safe_extract(archive, asset_root / str(spec["target"]))

    if key is not None:
        key_target = asset_root / "omnigibson.key"
        key_target.write_bytes(key + b"\n")
        key_target.chmod(0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        action="append",
        choices=tuple(COMPONENTS),
        dest="components",
        help="official asset component to materialize; repeat as needed",
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--case",
        choices=tuple(SELECTIVE_CASES),
        help="range-extract only the official asset closure for one lightweight case",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        help="user-obtained official omnigibson.key; required for the restricted behavior bundle",
    )
    args = parser.parse_args(argv)
    if args.case is not None:
        if args.components:
            parser.error("--case cannot be combined with --component")
        if args.key_file is None:
            parser.error("--case requires --key-file")
        manifest = materialize_selective_case(
            args.case,
            asset_root=args.asset_root,
            key_file=args.key_file,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    components = args.components or list(COMPONENTS)
    materialize(
        components,
        asset_root=args.asset_root,
        cache_root=args.cache_root,
        key_file=args.key_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
