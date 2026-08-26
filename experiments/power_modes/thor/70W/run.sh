#!/usr/bin/env bash
# thor @ 70W — verify mode is active (reboot after nvpmodel), then run.
# Usage: sudo -E env "PATH=$PATH" ./run.sh /path/to/bdd100k [num_clips]
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/../../common/run_power_mode.sh" thor 70W "$DIR" "$@"
