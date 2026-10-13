#!/bin/bash
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python tools/check_pfesa.py --gpu $GPU --ckpt "$BEST" 2>&1 | tee $LOGS/pfesa_check.log
