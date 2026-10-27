"""Synchronize CFF release metadata with the latest Zenodo deposit."""

import datetime
import json
import os
import re
import urllib.request
from pathlib import Path


def update_citation(record, concept_id, citation_path):
    metadata = record["metadata"]
    doi = record["conceptdoi"]
    if doi != f"10.5281/zenodo.{concept_id}":
        raise ValueError("Zenodo returned a record from a different deposit")

    # Both Rust and npm release tags are archived in the same Zenodo deposit.
    version = re.fullmatch(
        r"(?:eunoia-npm-)?v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)",
        metadata["version"],
    )
    if version is None:
        raise ValueError("Unrecognized Zenodo release version")

    release_date = datetime.date.fromisoformat(metadata["publication_date"])
    values = {
        "doi": doi,
        "version": version[1],
        "date-released": release_date.isoformat(),
    }

    original = citation_path.read_text()
    updated = original
    for field, value in values.items():
        # Limit edits to top-level scalars so author-written metadata stays intact.
        updated, count = re.subn(
            rf"^{field}:[^\r\n]*$",
            f"{field}: {json.dumps(value)}",
            updated,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"Expected exactly one top-level {field} field")

    if updated != original:
        citation_path.write_text(updated)
        print(f"Updated {citation_path} to version {version[1]} ({doi})")
    else:
        print(f"{citation_path} is already up to date")


def main():
    concept_id = os.environ["ZENODO_CONCEPT_ID"]
    if not re.fullmatch(r"[0-9]+", concept_id):
        raise ValueError("Invalid Zenodo concept ID")

    url = f"https://zenodo.org/api/records/{concept_id}/versions/latest"
    request = urllib.request.Request(
        url, headers={"User-Agent": "eunoia-citation-updater"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        record = json.load(response)

    update_citation(record, concept_id, Path("CITATION.cff"))


if __name__ == "__main__":
    main()
