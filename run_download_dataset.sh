#!/usr/bin/env bash
#
# download_dataset.sh
#
# Usage:
#   chmod +x download_dataset.sh
#   ./download_dataset.sh
#
# Edit AWS_URL if you need a different file.

set -euo pipefail

### Configuration ###
AWS_URL="https://olli-master-thesis.s3.eu-west-1.amazonaws.com/datasets.zip"
ZIP_NAME="datasets.zip"
DEST_DIR="data"

### Script ###
echo "→ Downloading ${AWS_URL} ..."
curl -L -o "${ZIP_NAME}" "${AWS_URL}"

echo "→ Unzipping to ${DEST_DIR}/ ..."
mkdir -p "${DEST_DIR}"
unzip -q "${ZIP_NAME}" -d "${DEST_DIR}"

echo "→ Cleaning up ..."
rm -f "${ZIP_NAME}"

echo "✓ Done! Extracted files are in '${DEST_DIR}/'."
