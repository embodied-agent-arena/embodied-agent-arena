#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cached_xvfb="${repo_root}/.cache/runtime/xvfb/Xvfb"
xvfb_bin="${RLBENCH_XVFB_BIN:-}"
xvfb_ld_library_path="${RLBENCH_XVFB_LD_LIBRARY_PATH:-}"
display_server="${RLBENCH_DISPLAY_SERVER:-xvfb}"
xorg_bin="${RLBENCH_XORG_BIN:-}"

if [[ -z "${xvfb_bin}" ]] && command -v Xvfb >/dev/null 2>&1; then
  xvfb_bin="$(command -v Xvfb)"
fi
if [[ -z "${xvfb_bin}" ]] && [[ -x "${cached_xvfb}" ]]; then
  xvfb_bin="${cached_xvfb}"
fi
if [[ -z "${xorg_bin}" ]] && [[ -x "/usr/lib/xorg/Xorg" ]]; then
  xorg_bin="/usr/lib/xorg/Xorg"
fi
if [[ "${display_server}" == "xorg_dummy" ]]; then
  if [[ -z "${xorg_bin}" ]] || [[ ! -x "${xorg_bin}" ]]; then
    echo "Xorg dummy display requested but Xorg is unavailable; set RLBENCH_XORG_BIN." >&2
    exit 127
  fi
elif [[ -z "${xvfb_bin}" ]] || [[ ! -x "${xvfb_bin}" ]]; then
  echo "Xvfb is unavailable; set RLBENCH_XVFB_BIN or prepare ${cached_xvfb}." >&2
  exit 127
fi
if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

job_number="${SLURM_JOB_ID:-$$}"
log="${RLBENCH_XVFB_LOG:-${TMPDIR:-/tmp}/xvfb-${USER:-codex}-${job_number}.log}"
xvfb_pid=""
display_number=""
display_mode=""
xvfb_env=()
xorg_config=""
if [[ -n "${xvfb_ld_library_path}" ]]; then
  xvfb_env=(env "LD_LIBRARY_PATH=${xvfb_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
fi
if [[ "${display_server}" == "xorg_dummy" ]]; then
  xorg_config="$(mktemp "${TMPDIR:-/tmp}/xorg-dummy-${USER:-codex}-${job_number}.XXXXXX.conf")"
  cat >"${xorg_config}" <<'EOF'
Section "ServerFlags"
    Option "AutoAddDevices" "false"
    Option "DontVTSwitch" "true"
    Option "AllowMouseOpenFail" "true"
EndSection

Section "Module"
    Load "glx"
EndSection

Section "Device"
    Identifier "dummy_device"
    Driver "dummy"
    VideoRam 256000
EndSection

Section "Monitor"
    Identifier "dummy_monitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
EndSection

Section "Screen"
    Identifier "dummy_screen"
    Device "dummy_device"
    Monitor "dummy_monitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1280x1024"
    EndSubSection
EndSection

Section "ServerLayout"
    Identifier "dummy_layout"
    Screen "dummy_screen"
EndSection
EOF
fi

if [[ -n "${RLBENCH_XVFB_DISPLAY:-}" ]]; then
  candidates=("${RLBENCH_XVFB_DISPLAY}")
else
  mapfile -t candidates < <(seq 99 198)
fi
for candidate in "${candidates[@]}"; do
  candidate_socket="/tmp/.X11-unix/X${candidate}"
  if [[ -e "/tmp/.X${candidate}-lock" ]] || [[ -S "${candidate_socket}" ]]; then
    continue
  fi
  for mode in unix tcp; do
    if [[ "${mode}" == "tcp" && "${RLBENCH_XVFB_TCP_FALLBACK:-1}" != "1" ]]; then
      continue
    fi
    if [[ "${display_server}" == "xorg_dummy" ]]; then
      xvfb_args=(
        ":${candidate}"
        -config "${xorg_config}"
        -logfile "${log}"
        -noreset
        +extension GLX
        +iglx
        -ac
      )
    else
      xvfb_args=(":${candidate}" -screen 0 1280x1024x24 +extension GLX +render +iglx -noreset -ac)
    fi
    if [[ "${mode}" == "unix" ]]; then
      xvfb_args+=(-nolisten tcp)
    else
      xvfb_args+=(-listen tcp -nolisten unix -nolisten local -nolisten inet6)
    fi
    if [[ "${display_server}" == "xorg_dummy" ]]; then
      "${xvfb_env[@]}" "${xorg_bin}" "${xvfb_args[@]}" >"${log}" 2>&1 &
    else
      "${xvfb_env[@]}" "${xvfb_bin}" "${xvfb_args[@]}" >"${log}" 2>&1 &
    fi
    candidate_pid=$!
    candidate_ready=0
    for _ in $(seq 1 20); do
      if [[ "${mode}" == "unix" ]]; then
        if [[ -S "${candidate_socket}" ]] && kill -0 "${candidate_pid}" >/dev/null 2>&1; then
          candidate_ready=1
          break
        fi
      elif command -v xdpyinfo >/dev/null 2>&1; then
        if DISPLAY="localhost:${candidate}" xdpyinfo >/dev/null 2>&1; then
          candidate_ready=1
          break
        fi
      elif kill -0 "${candidate_pid}" >/dev/null 2>&1; then
        sleep 0.5
        if kill -0 "${candidate_pid}" >/dev/null 2>&1; then
          candidate_ready=1
          break
        fi
      fi
      if ! kill -0 "${candidate_pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.05
    done
    if [[ "${candidate_ready}" == "1" ]]; then
      xvfb_pid="${candidate_pid}"
      display_number="${candidate}"
      display_mode="${mode}"
      break
    fi
    kill "${candidate_pid}" >/dev/null 2>&1 || true
    wait "${candidate_pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${xvfb_pid}" ]]; then
    break
  fi
done
if [[ -z "${xvfb_pid}" ]] || [[ -z "${display_number}" ]]; then
  cat "${log}" >&2 || true
  if [[ -e "/tmp/.X11-unix" ]]; then
    socket_dir_info="$(stat -c 'owner=%U:%G uid=%u gid=%g mode=%a type=%F' /tmp/.X11-unix 2>/dev/null || true)"
    if [[ -n "${socket_dir_info}" ]]; then
      echo "Xvfb socket directory diagnostic: /tmp/.X11-unix ${socket_dir_info}; unix X sockets require a root-owned sticky directory, otherwise Xvfb may fail before the benchmark starts." >&2
    fi
  else
    echo "Xvfb socket directory diagnostic: /tmp/.X11-unix is missing; Xvfb may be unable to create unix display sockets in this sandbox." >&2
  fi
  echo "Could not start Xvfb on any candidate display." >&2
  exit 1
fi

if [[ "${display_mode}" == "tcp" ]]; then
  display="localhost:${display_number}"
else
  display=":${display_number}"
fi
socket="/tmp/.X11-unix/X${display_number}"
cleanup() {
  kill "${xvfb_pid}" >/dev/null 2>&1 || true
  wait "${xvfb_pid}" >/dev/null 2>&1 || true
  if [[ -n "${xorg_config}" ]]; then
    rm -f "${xorg_config}"
  fi
}
trap cleanup EXIT INT TERM

if [[ "${display_mode}" == "unix" ]]; then
  for _ in $(seq 1 100); do
    if [[ -S "${socket}" ]]; then
      break
    fi
    if ! kill -0 "${xvfb_pid}" >/dev/null 2>&1; then
      cat "${log}" >&2 || true
      echo "Xvfb exited before display ${display} became ready." >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ ! -S "${socket}" ]]; then
    cat "${log}" >&2 || true
    echo "Timed out waiting for Xvfb display ${display}." >&2
    exit 1
  fi
else
  if command -v xdpyinfo >/dev/null 2>&1; then
    for _ in $(seq 1 100); do
      if DISPLAY="${display}" xdpyinfo >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "${xvfb_pid}" >/dev/null 2>&1; then
        cat "${log}" >&2 || true
        echo "Xvfb exited before TCP display ${display} became ready." >&2
        exit 1
      fi
      sleep 0.05
    done
    if ! DISPLAY="${display}" xdpyinfo >/dev/null 2>&1; then
      cat "${log}" >&2 || true
      echo "Timed out waiting for TCP Xvfb display ${display}." >&2
      exit 1
    fi
  elif ! kill -0 "${xvfb_pid}" >/dev/null 2>&1; then
    cat "${log}" >&2 || true
    echo "Xvfb exited before TCP display ${display} became ready." >&2
    exit 1
  else
    sleep 0.5
  fi
fi

export DISPLAY="${display}"
if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
  export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/runtime-${USER:-codex}"
fi
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" >/dev/null 2>&1 || true
"$@"
