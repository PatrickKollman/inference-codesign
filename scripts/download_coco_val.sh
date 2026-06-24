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
# convert_coco in ultralytics 8.4.70+ iterates over ALL json files in labels_dir,
# including captions_*.json which has no 'bbox' field and will crash.
# Isolate instances_val2017.json in a temp directory before converting.
TMP_INSTANCES="${DATA_DIR}_instances_tmp"
TMP_CONVERT="${DATA_DIR}_convert_tmp"
rm -rf "${TMP_INSTANCES}" "${TMP_CONVERT}"
mkdir -p "${TMP_INSTANCES}"
cp "${DATA_DIR}/annotations/instances_val2017.json" "${TMP_INSTANCES}/"

python -c "
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir='${TMP_INSTANCES}/',
    save_dir='${TMP_CONVERT}',
    use_segments=False,
    cls91to80=True,
)
"
mv "${TMP_CONVERT}/labels" "${DATA_DIR}/labels"
rm -rf "${TMP_INSTANCES}" "${TMP_CONVERT}"

N_LABELS=$(ls "${DATA_DIR}/labels/val2017/" | wc -l | tr -d ' ')
echo "=== ${N_LABELS} label files created (expected ~4952; 48 val2017 images have no bbox annotations) ==="
if [ "${N_LABELS}" -lt 4900 ]; then
    echo "ERROR: Label count unexpectedly low — conversion may have failed."
    exit 1
fi

echo "=== COCO val2017 ready at ${DATA_DIR} ==="
echo "Run: python scripts/fp32_eval.py --data-dir ${DATA_DIR}"
