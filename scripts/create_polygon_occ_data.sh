#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}

ROOT_PATH="./data/nuscenes"
CANBUS_PATH="./data/nuscenes"
INFO_DIR="./data/infos"
OCC_ROOT="./data/nuscenes/occ3d"
WORK_DIR="./data/polygon_occ_work"
OUT_DIR="./data/polygon_occ_infos"
KMEANS_DIR="./data/kmeans"
VERSION="v1.0-mini"
EXTRA_TAG="nuscenes"
TRAIN_INFO="${INFO_DIR}/nuscenes_infos_train.pkl"
VAL_INFO="${INFO_DIR}/nuscenes_infos_val.pkl"
ARRAY_KEY="semantics"
CLASS_MAPPING="./projects/configs/polygon_occ_class_mapping_v1.json"
BBOX_CLASS_MAPPING="./projects/configs/polygon_occ_bbox_mapping_v1.json"
PROJECT_AXIS="2"
NUM_POINTS="20"
MIN_PIXELS="4"
SIMPLIFY_RATIO="0.005"
DRIVEABLE_FILL_HOLES="1"
DRIVEABLE_CLOSE_KERNEL="3"
X_MIN="-40.0"
Y_MIN="-40.0"
X_STEP="0.4"
Y_STEP="0.4"
ROI_X_MIN="-15.0"
ROI_X_MAX="15.0"
ROI_Y_MIN="-30.0"
ROI_Y_MAX="30.0"
IGNORE_SOURCE_IDS="0,12,17"
GENERATE_ALL="0"
KEEP_WORK="0"

usage() {
    cat <<EOF
Usage: bash scripts/create_polygon_occ_data.sh [options]

One-click Polygon OCC data preparation pipeline:
  1. Build train/val Occ3D manifests from current info files
  2. Convert Occ3D semantics to token-keyed polygon_occ annotations
  3. Merge train/val polygon_occ annotation files
  4. Export split polygon_occ sidecar pkls for token-joined loading

Options:
  --root-path PATH          nuScenes root path (default: ${ROOT_PATH})
  --canbus PATH             nuScenes can bus path (default: ${CANBUS_PATH})
  --info-dir PATH           existing SparseDrive info dir (default: ${INFO_DIR})
  --occ-root PATH           Occ3D root dir (default: ${OCC_ROOT})
  --work-dir PATH           intermediate artifact dir (default: ${WORK_DIR})
  --out-dir PATH            output info dir with polygon occ (default: ${OUT_DIR})
  --kmeans-dir PATH         output dir for polygon occ anchors (default: ${KMEANS_DIR})
  --version VERSION         nuscenes_converter version (default: ${VERSION})
  --extra-tag TAG           output info prefix (default: ${EXTRA_TAG})
  --train-info PATH         train info pkl override
  --val-info PATH           val info pkl override
  --array-key KEY           occupancy array key (default: ${ARRAY_KEY})
  --class-mapping PATH      raw Occ3D id -> Polygon OCC id mapping (default: ${CLASS_MAPPING})
  --bbox-class-mapping PATH detection class name -> Polygon OCC id mapping (default: ${BBOX_CLASS_MAPPING})
  --project-axis N          occupancy projection axis (default: ${PROJECT_AXIS})
  --num-points N            polygon points per instance (default: ${NUM_POINTS})
  --min-pixels N            min connected-component area in pixels (default: ${MIN_PIXELS})
  --simplify-ratio FLOAT    contour simplify ratio (default: ${SIMPLIFY_RATIO})
  --disable-driveable-fill-holes  do not fill driveable_surface mask holes
  --driveable-close-kernel N close-kernel size for driveable_surface (default: ${DRIVEABLE_CLOSE_KERNEL})
  --x-min FLOAT             Occ3D metric x min before ROI clipping (default: ${X_MIN})
  --y-min FLOAT             Occ3D metric y min before ROI clipping (default: ${Y_MIN})
  --x-step FLOAT            Occ3D voxel size along x (default: ${X_STEP})
  --y-step FLOAT            Occ3D voxel size along y (default: ${Y_STEP})
  --roi-x-min FLOAT         SparseDrive ROI x min after conversion (default: ${ROI_X_MIN})
  --roi-x-max FLOAT         SparseDrive ROI x max after conversion (default: ${ROI_X_MAX})
  --roi-y-min FLOAT         SparseDrive ROI y min after conversion (default: ${ROI_Y_MIN})
  --roi-y-max FLOAT         SparseDrive ROI y max after conversion (default: ${ROI_Y_MAX})
  --ignore-source-ids IDS   comma-separated source ids to ignore (default: ${IGNORE_SOURCE_IDS})
  --generate-all            also generate merged polygon_occ_all.pkl
  --keep-work               keep intermediate files under ${WORK_DIR}
  -h, --help                show this message

Outputs:
  
  ${WORK_DIR}/occ_train_manifest.pkl and occ_val_manifest.pkl
  ${WORK_DIR}/polygon_occ_train.pkl and polygon_occ_val.pkl
  ${OUT_DIR}/polygon_occ_train.pkl
  ${OUT_DIR}/polygon_occ_val.pkl
  ${KMEANS_DIR}/kmeans_polygon_occ_100_${NUM_POINTS}.npy
  optional: polygon_occ_all.pkl when --generate-all is set
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-path)
            ROOT_PATH="$2"; shift 2 ;;
        --canbus)
            CANBUS_PATH="$2"; shift 2 ;;
        --info-dir)
            INFO_DIR="$2"
            TRAIN_INFO="${INFO_DIR}/nuscenes_infos_train.pkl"
            VAL_INFO="${INFO_DIR}/nuscenes_infos_val.pkl"
            shift 2 ;;
        --occ-root)
            OCC_ROOT="$2"; shift 2 ;;
        --work-dir)
            WORK_DIR="$2"; shift 2 ;;
        --out-dir)
            OUT_DIR="$2"; shift 2 ;;
        --kmeans-dir)
            KMEANS_DIR="$2"; shift 2 ;;
        --version)
            VERSION="$2"; shift 2 ;;
        --extra-tag)
            EXTRA_TAG="$2"; shift 2 ;;
        --train-info)
            TRAIN_INFO="$2"; shift 2 ;;
        --val-info)
            VAL_INFO="$2"; shift 2 ;;
        --array-key)
            ARRAY_KEY="$2"; shift 2 ;;
        --class-mapping)
            CLASS_MAPPING="$2"; shift 2 ;;
        --bbox-class-mapping)
            BBOX_CLASS_MAPPING="$2"; shift 2 ;;
        --project-axis)
            PROJECT_AXIS="$2"; shift 2 ;;
        --num-points)
            NUM_POINTS="$2"; shift 2 ;;
        --min-pixels)
            MIN_PIXELS="$2"; shift 2 ;;
        --simplify-ratio)
            SIMPLIFY_RATIO="$2"; shift 2 ;;
        --disable-driveable-fill-holes)
            DRIVEABLE_FILL_HOLES="0"; shift 1 ;;
        --driveable-close-kernel)
            DRIVEABLE_CLOSE_KERNEL="$2"; shift 2 ;;
        --x-min)
            X_MIN="$2"; shift 2 ;;
        --y-min)
            Y_MIN="$2"; shift 2 ;;
        --x-step)
            X_STEP="$2"; shift 2 ;;
        --y-step)
            Y_STEP="$2"; shift 2 ;;
        --roi-x-min)
            ROI_X_MIN="$2"; shift 2 ;;
        --roi-x-max)
            ROI_X_MAX="$2"; shift 2 ;;
        --roi-y-min)
            ROI_Y_MIN="$2"; shift 2 ;;
        --roi-y-max)
            ROI_Y_MAX="$2"; shift 2 ;;
        --ignore-source-ids)
            IGNORE_SOURCE_IDS="$2"; shift 2 ;;
        --generate-all)
            GENERATE_ALL="1"; shift 1 ;;
        --keep-work)
            KEEP_WORK="1"; shift 1 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1 ;;
    esac
done

mkdir -p "${WORK_DIR}" "${OUT_DIR}" "${KMEANS_DIR}"

echo "[1/7] Build train Occ3D manifest"
python3 tools/data_converter/build_occ3d_manifest.py \
    --info "${TRAIN_INFO}" \
    --occ-root "${OCC_ROOT}" \
    --output "${WORK_DIR}/occ_train_manifest.pkl" \
    --split train \
    --summary-path "${WORK_DIR}/occ_train_manifest_summary.pkl"

echo "[2/7] Build val Occ3D manifest"
python3 tools/data_converter/build_occ3d_manifest.py \
    --info "${VAL_INFO}" \
    --occ-root "${OCC_ROOT}" \
    --output "${WORK_DIR}/occ_val_manifest.pkl" \
    --split val \
    --summary-path "${WORK_DIR}/occ_val_manifest_summary.pkl"

echo "[3/7] Convert train Occ3D to Polygon OCC annotations"
python3 tools/data_converter/occ_to_polygon_occ.py \
    "${WORK_DIR}/occ_train_manifest.pkl" \
    "${WORK_DIR}/polygon_occ_train.pkl" \
    --input-type occupancy \
    --array-key "${ARRAY_KEY}" \
    --class-mapping "${CLASS_MAPPING}" \
    --bbox-class-mapping "${BBOX_CLASS_MAPPING}" \
    --project-axis "${PROJECT_AXIS}" \
    --num-points "${NUM_POINTS}" \
    --min-pixels "${MIN_PIXELS}" \
    --simplify-ratio "${SIMPLIFY_RATIO}" \
    --x-min "${X_MIN}" \
    --y-min "${Y_MIN}" \
    --x-step "${X_STEP}" \
    --y-step "${Y_STEP}" \
    --roi-x-min "${ROI_X_MIN}" \
    --roi-x-max "${ROI_X_MAX}" \
    --roi-y-min "${ROI_Y_MIN}" \
    --roi-y-max "${ROI_Y_MAX}" \
    --ignore-source-ids "${IGNORE_SOURCE_IDS}" \
    $(if [[ "${DRIVEABLE_FILL_HOLES}" == "1" ]]; then printf '%s ' "--driveable-fill-holes"; fi) \
    --driveable-close-kernel "${DRIVEABLE_CLOSE_KERNEL}" \
    --summary-path "${WORK_DIR}/polygon_occ_train_summary.pkl"

echo "[4/7] Convert val Occ3D to Polygon OCC annotations"
python3 tools/data_converter/occ_to_polygon_occ.py \
    "${WORK_DIR}/occ_val_manifest.pkl" \
    "${WORK_DIR}/polygon_occ_val.pkl" \
    --input-type occupancy \
    --array-key "${ARRAY_KEY}" \
    --class-mapping "${CLASS_MAPPING}" \
    --bbox-class-mapping "${BBOX_CLASS_MAPPING}" \
    --project-axis "${PROJECT_AXIS}" \
    --num-points "${NUM_POINTS}" \
    --min-pixels "${MIN_PIXELS}" \
    --simplify-ratio "${SIMPLIFY_RATIO}" \
    --x-min "${X_MIN}" \
    --y-min "${Y_MIN}" \
    --x-step "${X_STEP}" \
    --y-step "${Y_STEP}" \
    --roi-x-min "${ROI_X_MIN}" \
    --roi-x-max "${ROI_X_MAX}" \
    --roi-y-min "${ROI_Y_MIN}" \
    --roi-y-max "${ROI_Y_MAX}" \
    --ignore-source-ids "${IGNORE_SOURCE_IDS}" \
    $(if [[ "${DRIVEABLE_FILL_HOLES}" == "1" ]]; then printf '%s ' "--driveable-fill-holes"; fi) \
    --driveable-close-kernel "${DRIVEABLE_CLOSE_KERNEL}" \
    --summary-path "${WORK_DIR}/polygon_occ_val_summary.pkl"

if [[ "${GENERATE_ALL}" == "1" ]]; then
    echo "[5/7] Merge train/val Polygon OCC annotations"
    python3 tools/data_converter/merge_polygon_occ.py \
        "${WORK_DIR}/polygon_occ_train.pkl" \
        "${WORK_DIR}/polygon_occ_val.pkl" \
        --output "${WORK_DIR}/polygon_occ_all.pkl" \
        --summary-path "${WORK_DIR}/polygon_occ_all_summary.pkl"
else
    echo "[5/7] Skip merged polygon_occ_all.pkl (default behavior)"
fi

echo "[6/7] Export split Polygon OCC annotation sidecars"
cp "${WORK_DIR}/polygon_occ_train.pkl" "${OUT_DIR}/polygon_occ_train.pkl"
cp "${WORK_DIR}/polygon_occ_val.pkl" "${OUT_DIR}/polygon_occ_val.pkl"
if [[ "${GENERATE_ALL}" == "1" ]]; then
    cp "${WORK_DIR}/polygon_occ_all.pkl" "${OUT_DIR}/polygon_occ_all.pkl"
fi

echo "[7/7] Build Polygon OCC anchors"
python3 tools/kmeans/kmeans_polygon_occ.py \
    --input "${OUT_DIR}/polygon_occ_train.pkl" \
    --output "${KMEANS_DIR}/kmeans_polygon_occ_100_${NUM_POINTS}.npy"

cat <<EOF

Polygon OCC data preparation finished.
EOF

if [[ "${GENERATE_ALL}" == "1" ]]; then
cat <<EOF
  Optional merged sidecar exported: ${OUT_DIR}/polygon_occ_all.pkl
EOF
fi

cat <<EOF

Output sidecars:
  ${OUT_DIR}
EOF

if [[ "${KEEP_WORK}" == "1" ]]; then
cat <<EOF

Intermediate work directory preserved:
  ${WORK_DIR}

Recommended checks:
  python3 - <<'PY'
import mmcv
print(mmcv.load('${WORK_DIR}/polygon_occ_train_summary.pkl'))
print(mmcv.load('${WORK_DIR}/polygon_occ_val_summary.pkl'))
PY
EOF
else
    echo
    echo "Cleaning intermediate work directory: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
fi
