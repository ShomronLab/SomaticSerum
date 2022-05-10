#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodelist=compute-0-2
#SBATCH --job-name=IndexName
#SBATCH --output=.%x_%j_%a.log 
pwd; hostname; date
set -e
module load miniconda/miniconda-py38_4.9.2-environmentally
conda activate SomaticExtractEnv

THREADS=$((`grep ^processor /proc/cpuinfo | wc -l`))

NORMAL=$1
OUT=$2
NAME=$3
TUMOR=$4
PLASMA=$5


if [[ -f ${NORMAL}.bai ]];then
    echo "${NORMAL}.bai exists"
else
    echo "samtools index -@ $THREADS $NORMAL"
    samtools index -@ $THREADS $NORMAL &
fi
if [[ -f ${TUMOR}.bai ]];then
    echo "${TUMOR}.bai exists"
else
    echo "samtools index -@ $THREADS $TUMOR"
    samtools index -@ $THREADS $TUMOR &
fi
if [[ -f ${PLASMA}.bai ]];then
    echo "${PLASMA}.bai exists"
else
    echo "samtools index -@ $THREADS $PLASMA"
    samtools index -@ $THREADS $PLASMA &
fi
wait

gatk GetSampleName \
    -I ${NORMAL} \
    -O ${OUT}/${NAME}.txt
