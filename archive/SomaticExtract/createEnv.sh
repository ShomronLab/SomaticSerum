#!/bin/bash
set -ex
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

module load miniconda/miniconda-py38_4.9.2-environmentally
conda env create -f ${SCRIPT_DIR}/refs/SomaticExtract_env.yml
