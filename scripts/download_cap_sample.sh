#!/bin/sh
# Download a small CAP volume for local indexing (Arkansas vol 14 example).
# See https://case.law/docs/ for bulk download documentation.

set -e
DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR/downloads"
cd "$DATA_DIR/downloads"

echo "Downloading Arkansas reporter volume 14 from static.case.law..."
curl -fL -o ark_14.zip "https://static.case.law/ark/14.zip"
unzip -o ark_14.zip -d ark_14
echo "Extract cases JSON into data/cases/ and run: python scripts/build_cap_index.py"
