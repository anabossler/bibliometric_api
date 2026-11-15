"""Upload every file of a folder to a Zenodo draft, without publishing it.

The draft is either a new record or a new version of a published record
(--new-version RECORD_ID). Metadata are read from a JSON file in the Zenodo
deposit format, and a DOI is pre-reserved so that it can be cited before
publication. The draft is left for review and publication on the website,
because publishing cannot be undone.

The token is read from ZENODO_TOKEN in the .env file given by --env and is
never printed. It needs the scopes deposit:write and deposit:actions.
--sandbox uses sandbox.zenodo.org, which needs a token created there.
--dry-run only lists what would be uploaded.

Usage:
    python zenodo_upload.py FOLDER --metadata META.json [--new-version RECORD_ID]
                            [--sandbox] [--dry-run] [--env PATH]
"""
import argparse
import json
import sys
from pathlib import Path

REQUIRED = ("upload_type", "title", "creators", "description")


def read_token(env):
    path = Path(env).expanduser()
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ZENODO_TOKEN":
            return value.strip().strip('"').strip("'")
    sys.exit(f"ZENODO_TOKEN not found in {path}")


def check(response, what):
    if response.status_code >= 400:
        sys.exit(f"{what}: HTTP {response.status_code} {response.text[:300]}")
    return response.json() if response.content else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--new-version", dest="new_version")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env", default="~/Desktop/openalex/.env")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.folder).iterdir() if p.is_file())
    if not files:
        sys.exit(f"no files in {args.folder}")
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if not meta.get(k)]
    if missing:
        sys.exit(f"metadata without {missing}")
    meta["prereserve_doi"] = True
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 1e6:.2f} MB | title: {meta['title']}")
    if args.dry_run:
        for p in files:
            print(f"  {p.name} ({p.stat().st_size} bytes)")
        return

    import requests
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {read_token(args.env)}"

    if args.new_version:
        parent = check(s.post(f"{base}/deposit/depositions/{args.new_version}/actions/newversion"),
                       "new version")
        draft = check(s.get(parent["links"]["latest_draft"]), "draft")
        names = {p.name for p in files}
        for f in draft.get("files", []):
            if f["filename"] in names:
                check(s.delete(f["links"]["self"]), f"delete {f['filename']}")
            else:
                print(f"kept from the previous version (delete it on the website if obsolete): "
                      f"{f['filename']}")
    else:
        draft = check(s.post(f"{base}/deposit/depositions", json={}), "create draft")

    bucket = draft["links"]["bucket"]
    for p in files:
        with p.open("rb") as fh:
            check(s.put(f"{bucket}/{p.name}", data=fh), f"upload {p.name}")
        print(f"uploaded {p.name}")
    draft = check(s.put(draft["links"]["self"], json={"metadata": meta}), "metadata")
    doi = (draft["metadata"].get("prereserve_doi") or {}).get("doi")
    print(f"draft: {draft['links']['html']}")
    print(f"reserved DOI: {doi or 'not returned; reserve it on the website'}")
    print("Not published. Review the draft on the website and publish it there.")


if __name__ == "__main__":
    main()
