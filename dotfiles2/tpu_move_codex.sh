#!/usr/bin/env bash
set -euo pipefail

source_dir="${HOME}/.codex"
target_dir="/mnt/carles/.codex"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="${HOME}/.codex.old.${timestamp}"

mkdir -p /mnt/carles
mkdir -p "${target_dir}"

if [ -L "${source_dir}" ]; then
    current_target="$(readlink -f "${source_dir}")"
    if [ "${current_target}" = "${target_dir}" ]; then
        echo "~/.codex already points to ${target_dir}"
        exit 0
    fi
    echo "~/.codex is a symlink to ${current_target}, not changing it"
    exit 1
fi

if [ -d "${source_dir}" ]; then
    rsync -a "${source_dir}/" "${target_dir}/"
    mv "${source_dir}" "${backup_dir}"
fi

ln -s "${target_dir}" "${source_dir}"

if [ -d "${backup_dir}" ]; then
    rsync -a "${backup_dir}/" "${target_dir}/"
fi

echo "Codex home moved to ${target_dir}"
if [ -d "${backup_dir}" ]; then
    echo "Backup kept at ${backup_dir}"
fi
echo "Restart Codex after this shell session to ensure all new writes use ${target_dir}"
