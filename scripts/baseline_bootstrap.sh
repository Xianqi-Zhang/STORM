#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STORM_REPO="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
STORM_ROOT="$(CDPATH= cd -- "${STORM_REPO}/.." && pwd)"
REPOS_DIR="${STORM_ROOT}/repos"
DATASETS_DIR="${STORM_ROOT}/DATASETS"
INTERACT_DIR="${REPOS_DIR}/InterAct"
INTERMIMIC_DIR="${REPOS_DIR}/InterMimic"

MODE="${1:-check}"

ok() {
  printf '[OK] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
}

info() {
  printf '[INFO] %s\n' "$1"
}

find_omomo_raw_src() {
  local candidates=(
    "${DATASETS_DIR}/OMOMO/data"
    "${DATASETS_DIR}/OMOMO/raw"
  )
  local cand
  for cand in "${candidates[@]}"; do
    if [[ -d "${cand}" ]]; then
      if [[ -f "${cand}/train_diffusion_manip_seq_joints24.p" && -f "${cand}/test_diffusion_manip_seq_joints24.p" ]]; then
        printf '%s' "${cand}"
        return 0
      fi
    fi
  done

  printf ''
}

find_intercap_raw_src() {
  local candidates=(
    "${DATASETS_DIR}/InterCap/raw"
    "${DATASETS_DIR}/InterCap"
  )
  local cand
  for cand in "${candidates[@]}"; do
    if [[ -d "${cand}" ]]; then
      if [[ -n "$(find "${cand}" -maxdepth 4 -name 'res.pkl' -print -quit 2>/dev/null)" ]]; then
        printf '%s' "${cand}"
        return 0
      fi
    fi
  done

  printf ''
}

ensure_symlink() {
  local link_path="$1"
  local target_path="$2"

  mkdir -p "$(dirname -- "${link_path}")"

  if [[ -L "${link_path}" ]]; then
    ln -sfn "${target_path}" "${link_path}"
    ok "Updated symlink ${link_path} -> ${target_path}"
    return 0
  fi

  if [[ -e "${link_path}" ]]; then
    warn "Skip ${link_path}: path exists and is not a symlink."
    return 1
  fi

  ln -s "${target_path}" "${link_path}"
  ok "Created symlink ${link_path} -> ${target_path}"
}

print_check_report() {
  info "Workspace root: ${STORM_ROOT}"

  if [[ -d "${INTERACT_DIR}" ]]; then
    ok "InterAct repo found: ${INTERACT_DIR}"
  else
    warn "InterAct repo not found: ${INTERACT_DIR}"
  fi

  if [[ -d "${INTERMIMIC_DIR}" ]]; then
    ok "InterMimic repo found: ${INTERMIMIC_DIR}"
  else
    warn "InterMimic repo not found: ${INTERMIMIC_DIR}"
  fi

  local omomo_raw_src
  omomo_raw_src="$(find_omomo_raw_src)"
  if [[ -n "${omomo_raw_src}" ]]; then
    ok "OMOMO raw data ready: ${omomo_raw_src}"
  else
    warn "OMOMO raw data not ready."
    if [[ -f "${DATASETS_DIR}/OMOMO/data.tar.gz" ]]; then
      info "Extract with: tar -xzf ${DATASETS_DIR}/OMOMO/data.tar.gz -C ${DATASETS_DIR}/OMOMO"
    else
      warn "Missing archive: ${DATASETS_DIR}/OMOMO/data.tar.gz"
    fi
  fi

  if [[ -d "${DATASETS_DIR}/OMOMO/omomo_text_anno_json_data" ]]; then
    ok "OMOMO text annotations found: ${DATASETS_DIR}/OMOMO/omomo_text_anno_json_data"
  else
    warn "OMOMO text annotations not found."
    if [[ -f "${DATASETS_DIR}/OMOMO/omomo_text_anno.zip" ]]; then
      info "Unzip with: unzip -o ${DATASETS_DIR}/OMOMO/omomo_text_anno.zip -d ${DATASETS_DIR}/OMOMO"
    else
      warn "Missing archive: ${DATASETS_DIR}/OMOMO/omomo_text_anno.zip"
    fi
  fi

  local intercap_raw_src
  intercap_raw_src="$(find_intercap_raw_src)"
  if [[ -n "${intercap_raw_src}" ]]; then
    ok "InterCap raw data ready: ${intercap_raw_src}"
  else
    warn "InterCap raw data not detected (expected files named res.pkl)."
  fi

  if [[ -f "${INTERACT_DIR}/models/smplx/SMPLX_MALE.npz" ]]; then
    ok "InterAct SMPLX model found."
  else
    warn "InterAct SMPLX model missing at ${INTERACT_DIR}/models/smplx/SMPLX_MALE.npz"
  fi

  if [[ -f "${INTERACT_DIR}/models/smplh/male/model.npz" ]]; then
    ok "InterAct SMPLH model found."
  else
    warn "InterAct SMPLH model missing at ${INTERACT_DIR}/models/smplh/male/model.npz"
  fi

  if [[ -L "${INTERACT_DIR}/data/omomo/raw" ]]; then
    ok "InterAct OMOMO link exists: ${INTERACT_DIR}/data/omomo/raw"
  else
    warn "InterAct OMOMO link missing: ${INTERACT_DIR}/data/omomo/raw"
  fi

  if [[ -L "${INTERACT_DIR}/data/intercap/raw" ]]; then
    ok "InterAct InterCap link exists: ${INTERACT_DIR}/data/intercap/raw"
  else
    warn "InterAct InterCap link missing: ${INTERACT_DIR}/data/intercap/raw"
  fi

  if [[ -d "${INTERACT_DIR}/simulation/intermimic/InterAct/omomo" ]]; then
    ok "InterAct->InterMimic converted motions found."
  else
    warn "No converted motions yet at ${INTERACT_DIR}/simulation/intermimic/InterAct/omomo"
  fi
}

run_link() {
  local omomo_raw_src
  omomo_raw_src="$(find_omomo_raw_src)"
  if [[ -z "${omomo_raw_src}" ]]; then
    warn "Cannot link OMOMO: source data is not ready. Run '${0} check' first."
  else
    ensure_symlink "${INTERACT_DIR}/data/omomo/raw" "${omomo_raw_src}" || true

    if [[ -d "${DATASETS_DIR}/OMOMO/omomo_text_anno_json_data" && ! -e "${omomo_raw_src}/omomo_text_anno_json_data" ]]; then
      ln -s "${DATASETS_DIR}/OMOMO/omomo_text_anno_json_data" "${omomo_raw_src}/omomo_text_anno_json_data"
      ok "Linked OMOMO text annotations into ${omomo_raw_src}"
    fi
  fi

  local intercap_raw_src
  intercap_raw_src="$(find_intercap_raw_src)"
  if [[ -z "${intercap_raw_src}" ]]; then
    warn "Cannot link InterCap: raw data not detected."
  else
    ensure_symlink "${INTERACT_DIR}/data/intercap/raw" "${intercap_raw_src}" || true
  fi

  local converted_omomo="${INTERACT_DIR}/simulation/intermimic/InterAct/omomo"
  if [[ -d "${converted_omomo}" ]]; then
    ensure_symlink "${INTERMIMIC_DIR}/InterAct/OMOMO_new" "${converted_omomo}" || true
  else
    info "Skip InterMimic motion link: converted OMOMO data not found yet (${converted_omomo})."
  fi
}

case "${MODE}" in
  check)
    print_check_report
    ;;
  link)
    run_link
    ;;
  *)
    echo "Usage: $0 [check|link]"
    exit 1
    ;;
esac
