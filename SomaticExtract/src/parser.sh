#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodelist=compute-0-2
#SBATCH --job-name=SomaticExtract
#SBATCH --output=.%x_%j.log 
pwd; hostname; date
set -e
module load miniconda/miniconda-py38_4.9.2-environmentally
conda activate SomaticExtractEnv

OUT=$1
NAME=$2
REF=$3
mutectParser=$4
PLASMA=$5

echo -n > ${OUT}/input_variant_files.list
ls ${OUT}/vcf/filtered*vcf.gz >> ${OUT}/input_variant_files.list

gatk MergeVcfs \
    -I ${OUT}/input_variant_files.list \
    -O ${OUT}/filtered.${NAME}.vcf.gz

python $mutectParser ${OUT}/filtered.${NAME}.vcf.gz $PLASMA mutect $NAME $OUT

# echo "samtools view $PLASMA | grep -f somatic_reads.${NAME}.txt > ${OUT}/somatic_reads.${NAME}.fastq"
# samtools view $PLASMA | grep -f ${OUT}/somatic_reads.${NAME}.txt > ${OUT}/somatic_reads.${NAME}.fastq

gatk FilterSamReads \
    -I $PLASMA \
    -O ${OUT}/somatic_reads.${NAME}.bam \
    --READ_LIST_FILE ${OUT}/somatic_reads.${NAME}.txt \
    --FILTER includeReadList \
    --CREATE_INDEX &

gatk FilterSamReads \
    -I $PLASMA \
    -O ${OUT}/normal_reads.${NAME}.bam \
    --READ_LIST_FILE ${OUT}/normal_reads.${NAME}.txt \
    --FILTER includeReadList \
    --CREATE_INDEX &

wait