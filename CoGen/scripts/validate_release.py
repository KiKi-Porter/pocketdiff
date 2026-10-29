from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"
CHECKSUMS = ROOT / "SHA256SUMS"
REPORT = ROOT / "validation_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    artifacts = []
    listed_paths = set()
    errors = []
    for line in CHECKSUMS.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*").removeprefix("./")
        if relative in listed_paths:
            errors.append(f"duplicate checksum entry: {relative}")
        listed_paths.add(relative)
        artifacts.append({"path": relative, "sha256": digest})

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if (path.is_file() or path.is_symlink()) and path not in (CHECKSUMS, REPORT)
    }
    for relative in sorted(actual_paths - listed_paths):
        errors.append(f"unlisted artifact: {relative}")
    for relative in sorted(listed_paths - actual_paths):
        errors.append(f"stale checksum entry: {relative}")

    checked = []
    for entry in artifacts:
        path = ROOT / entry["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {entry['path']}")
            continue
        actual = _sha256(path)
        if actual != entry["sha256"]:
            errors.append(f"SHA256 mismatch: {entry['path']}")
        checked.append({"path": entry["path"], "sha256_ok": actual == entry["sha256"]})

    if errors:
        report = {"passed": False, "errors": errors, "artifacts": checked}
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    python_files = sorted(
        path
        for tree in (ROOT / "pocketdiff", ROOT / "ligand_generator", ROOT / "scripts")
        for path in tree.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    syntax_errors = []
    for path in python_files:
        try:
            source = path.read_text()
            compile(source, str(path), "exec")
        except Exception as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")
    if syntax_errors:
        errors.extend(syntax_errors)

    report = {
        "release_root": str(ROOT),
        "release_id": manifest["release_id"],
        "artifacts_checked": len(checked),
        "all_hashes_valid": all(item["sha256_ok"] for item in checked),
        "python_files_syntax_checked": len(python_files),
        "syntax_errors": syntax_errors,
        "errors": errors,
        "passed": not errors,
        "scope": "exact artifact inventory, SHA-256 integrity, and Python syntax only; full training and 1000-step sampling are not run",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
