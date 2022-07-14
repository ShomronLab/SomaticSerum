#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodelist=compute-0-2
#SBATCH --job-name=Mutect2
#SBATCH --array=1-25 
#SBATCH --output=.%x_%A_%a.log 
pwd; hostname; date
set -e
module load miniconda/miniconda-py38_4.9.2-environmentally
conda activate SomaticExtractEnv

REF=$1
TUMOR=$2
NORMAL=$3
gnomad=$4
OUT=$5
NAME=$6
PoN=$7

normal_name=$(head -n 1 ${OUT}/${NAME}.txt)

chrs=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 MT X Y ) 
CHR=${chrs[$SLURM_ARRAY_TASK_ID]}

gatk Mutect2 \
    -R $REF \
    -L $CHR \
    -I $TUMOR \
    -I $NORMAL \
    -normal $normal_name \
    --germline-resource ${gnomad} \
    --panel-of-normals ${PoN} \
    -O ${OUT}/vcf/${NAME}.${CHR}.vcf.gz

gatk FilterMutectCalls \
    -R $REF \
    -V ${OUT}/vcf/${NAME}.${CHR}.vcf.gz \
    -O ${OUT}/vcf/filtered.${NAME}.${CHR}.vcf.gz 

