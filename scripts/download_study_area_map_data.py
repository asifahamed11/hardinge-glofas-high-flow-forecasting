"""Download and verify public map data used by the study-area figure."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = REPOSITORY_ROOT / "data" / "external" / "map_sources"


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    sha256: str


ARCHIVES = (
    Archive(
        "ne_10m_admin_0_countries.zip",
        "https://naturalearth.s3.amazonaws.com/10m_cultural/"
        "ne_10m_admin_0_countries.zip",
        "ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6",
    ),
    Archive(
        "hybas_as_lev06_v1c.zip",
        "https://data.hydrosheds.org/file/hydrobasins/standard/"
        "hybas_as_lev06_v1c.zip",
        "7131e675b93e4e0d1185fa597c966026c8fc25cb4616ab0622505fa3caf2ce2c",
    ),
    Archive(
        "HydroRIVERS_v10_as_shp.zip",
        "https://data.hydrosheds.org/file/HydroRIVERS/"
        "HydroRIVERS_v10_as_shp.zip",
        "29780b0a75f90024f22e7e2029e5e3045f7325cda0528db65c5cc4c864b98525",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(archive: Archive) -> Path:
    destination = DESTINATION / archive.filename
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and file_sha256(destination) == archive.sha256:
        print(f"Verified existing archive: {archive.filename}")
        return destination

    partial.unlink(missing_ok=True)
    print(f"Downloading {archive.filename}")
    with (
        urllib.request.urlopen(archive.url, timeout=120) as response,  # noqa: S310
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1024 * 1024)

    actual = file_sha256(partial)
    if actual != archive.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {archive.filename}: {actual}"
        )
    partial.replace(destination)
    return destination


def extract(archive_path: Path) -> None:
    destination = DESTINATION / archive_path.stem
    if destination.exists() and any(destination.rglob("*.shp")):
        print(f"Already extracted: {archive_path.name}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    print(f"Extracted: {archive_path.name}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        extract(download(archive))
    print(f"Map sources are ready in {DESTINATION}")


if __name__ == "__main__":
    main()
