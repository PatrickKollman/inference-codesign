#!/usr/bin/env bash
# Download COCO val2017 images + annotations to data/coco/.
# Run once on the pod after setup_runpod.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/coco"

mkdir -p "${DATA_DIR}/images"
mkdir -p "${DATA_DIR}/annotations"

echo "=== Downloading COCO val2017 images (~778 MB) ==="
wget -q --show-progress \
    -O "${DATA_DIR}/val2017.zip" \
    "http://images.cocodataset.org/zips/val2017.zip"
unzip -q "${DATA_DIR}/val2017.zip" -d "${DATA_DIR}/images/"
rm "${DATA_DIR}/val2017.zip"

echo "=== Downloading COCO annotations (~252 MB) ==="
wget -q --show-progress \
    -O "${DATA_DIR}/annotations.zip" \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip -q "${DATA_DIR}/annotations.zip" -d "${DATA_DIR}/"
rm "${DATA_DIR}/annotations.zip"

N_IMAGES=$(ls "${DATA_DIR}/images/val2017/" | wc -l | tr -d ' ')
echo "=== ${N_IMAGES} images extracted (expected 5000) ==="
if [ "${N_IMAGES}" -ne 5000 ]; then
    echo "ERROR: Image count mismatch — download may be incomplete."
    exit 1
fi

ANNOT="${DATA_DIR}/annotations/instances_val2017.json"
if [ ! -f "${ANNOT}" ]; then
    echo "ERROR: Annotation file not found: ${ANNOT}"
    exit 1
fi

echo "=== Converting COCO JSON annotations to YOLO label format ==="
# convert_coco creates a new directory at save_dir (incrementing if it exists).
# Use a dedicated temp dir and move labels into place to avoid path conflicts.
TMP_CONVERT="${DATA_DIR}_convert_tmp"
rm -rf "${TMP_CONVERT}"
python -c "
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir='${DATA_DIR}/annotations/',
    save_dir='${TMP_CONVERT}',
    use_segments=False,
    cls91to80=True,
)
"
mv "${TMP_CONVERT}/labels" "${DATA_DIR}/labels"
rm -rf "${TMP_CONVERT}"

N_LABELS=\$(ls "${DATA_DIR}/labels/val2017/" | wc -l | tr -d ' ')
echo "=== \${N_LABELS} label files created (expected 5000) ==="
if [ "\${N_LABELS}" -ne 5000 ]; then
    echo "ERROR: Label count mismatch — conversion may have failed."
    exit 1
fi

echo "=== COCO val2017 ready at ${DATA_DIR} ==="
echo "Run: python scripts/day3_eval_fp32.py --data-dir ${DATA_DIR}"
