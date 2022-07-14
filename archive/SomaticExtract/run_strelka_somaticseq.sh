#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=compute-0-1
#SBATCH --job-name=SomEx2
#SBATCH --array=1-7
#SBATCH --output=.%x_%A_%a.log
pwd; hostname; date
set -e

THREADS=$((`grep ^processor /proc/cpuinfo | wc -l`))
THREADS=24
. "/data/hadasvol/tools/miniconda3/etc/profile.d/conda.sh"
conda activate somaticseq2


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SCRIPT_DIR=/data/hadasvol/projects/cancer_plasma/somaticserum/SomaticExtract
ExtractSNVs=${SCRIPT_DIR}/src/ExtractSNVs.py

# s=$1
s=`sed -n ${SLURM_ARRAY_TASK_ID}p incompl.txt`
mkdir -p LUAD${s}
BAMS=/data/hadasvol/projects/cancer_plasma/LUAD${s}
REF=/data/hadasvol/refs/HumanG1Kv37/human_g1k_v37.fasta
INCLU=/data/hadasvol/refs/HumanG1Kv37/human_g1k_v37-whitelist.bed

/data/hadasvol/tools/strelka-2.9.10.centos6_x86_64/bin/configureStrelkaSomaticWorkflow.py \
    --normalBam ${BAMS}/normal-LUAD-${s}.bam \
    --tumorBam ${BAMS}/tumor-LUAD-${s}.bam \
    --referenceFasta ${REF} \
    --runDir LUAD${s}/strelka2
LUAD${s}/strelka2/runWorkflow.py -m local -j 20

somaticseq_parallel.py \
    --output-directory  LUAD${s} \
    --genome-reference  ${REF} \
    --inclusion-region  ${INCLU} \
    --algorithm         xgboost \
    --threads           20 \
paired \
    --tumor-bam-file    ${BAMS}/tumor-LUAD-${s}.bam \
    --normal-bam-file   ${BAMS}/normal-LUAD-${s}.bam \
    --mutect2-vcf       ${BAMS}/results_LUAD${s}/filtered.LUAD${s}.vcf.gz \
    --strelka-snv       LUAD${s}/strelka2/results/variants/somatic.snvs.vcf.gz \
    --strelka-indel     LUAD${s}/strelka2/results/variants/somatic.indels.vcf.gz


conda activate py3.6

orig_palsma=/data/hadasvol/projects/cancer_plasma/LUAD${s}/pre-LUAD-${s}.bam
if test -f "$orig_palsma"; then
        echo "$orig_palsma exists."
fi
samtools collate -@ 96 -o LUAD${s}/pre-LUAD${s}.collate.bam $orig_palsma
samtools fastq -1 LUAD${s}/pre-LUAD${s}.R1.fq.gz -2 LUAD${s}/pre-LUAD${s}.R2.fq.gz -@ 96 LUAD${s}/pre-LUAD${s}.collate.bam
bbmerge-auto.sh in1=LUAD${s}/pre-LUAD${s}.R1.fq.gz in2=LUAD${s}/pre-LUAD${s}.R2.fq.gz out=LUAD${s}/pre-LUAD${s}.bb.fq outu1=LUAD${s}/pre-LUAD${s}.bb.R1.fq outu2=LUAD${s}/pre-LUAD${s}.bb.R2.fq maxstrict=f ihist=LUAD${s}/ihist.txt

bwa mem -t $THREADS -R "@RG\tID:LUAD${s}\tPL:ILLUMINA\tSM:LUAD${s}\tLB:Seq" $REF LUAD${s}/pre-LUAD${s}.bb.R1.fq LUAD${s}/pre-LUAD${s}.bb.R2.fq | samblaster -M --ignoreUnmated | samtools view -@ $THREADS -Shb - | sambamba sort --tmpdir=. -t $THREADS /dev/stdin -o LUAD${s}/pre-LUAD${s}.pe.srt.mdup.bam
bwa mem -t $THREADS -R "@RG\tID:LUAD${s}\tPL:ILLUMINA\tSM:LUAD${s}\tLB:Seq" $REF LUAD${s}/pre-LUAD${s}.bb.fq | samblaster -M --ignoreUnmated | samtools view -@ $THREADS -Shb - | sambamba sort --tmpdir=./LUAD${s} -t $THREADS /dev/stdin -o LUAD${s}/pre-LUAD${s}.se.srt.mdup.bam
samtools merge -@ $THREADS -O BAM -o LUAD${s}/pre-LUAD${s}.srt.mdup.bam LUAD${s}/pre-LUAD${s}.se.srt.mdup.bam LUAD${s}/pre-LUAD${s}.pe.srt.mdup.bam
samtools index -@ $THREADS LUAD${s}/pre-LUAD${s}.srt.mdup.bam
reformat.sh in=LUAD${s}/pre-LUAD${s}.srt.mdup.bam out=LUAD${s}/pre-LUAD${s}.srt.mdup.refrmt.bam sam=1.4
samtools index -@ $THREADS LUAD${s}/pre-LUAD${s}.srt.mdup.refrmt.bam
python ${ExtractSNVs} LUAD${s}/Consensus.sSNV.vcf LUAD${s}/pre-LUAD${s}.srt.mdup.refrmt.bam LUAD${s} LUAD${s} False

#rm LUAD${s}/pre-LUAD${s}.bb.R1.fq LUAD${s}/pre-LUAD${s}.bb.R2.fq LUAD${s}/pre-LUAD${s}.bb.fq LUAD${s}/pre-LUAD${s}.se.srt.mdup.bam LUAD${s}/pre-LUAD${s}.pe.srt.mdup.bam


# conda activate py3.8
# fastqc LUAD${s}/somatic.SNVs.srt.bam
# fastqc LUAD${s}/normal.SNVs.srt.bam
# samtools stats LUAD${s}/somatic.SNVs.srt.bam > LUAD${s}/somatic.SNVs.srt.bam.stats
# samtools stats LUAD${s}/normal.SNVs.srt.bam > LUAD${s}/normal.SNVs.srt.bam.stats
# multiqc -o LUAD${s} -n luad${s}_report LUAD${s}/
