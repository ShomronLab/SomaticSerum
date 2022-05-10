# Install
## If running on nshomron2:
./createEnv.sh
This will initialize a conda environment with all needed tools and dependencies
## Runing wild:
Make sure the below dependencies are installed and in path:
gatk4=4.2.3.0
python=3.6.13
    pysam=0.17.0
    pandas=1.1.5
    pyvcf=0.6.8
samtools=1.14

# Execute
Execute the main bash script SomaticExtract.sh with the following arguments
    -n|--normal path to normal bam sample
    -t|--tumor pathe to tumor bam sample
    -p|--plasma path to plasma bam sample
    -o|--OUT_PREFIX output prefix
    -r|--REFERENCE ref b37 fasta indexed - default:/scratch/hadasvol/refs/HumanG1Kv37/human_g1k_v37.fasta
    -g|--germline Population vcf of germline sequencing containing allele fractions - default:/scratch/hadasvol/refs/HumanG1Kv37/af-only-gnomad.raw.sites.b37.vcf.gz
    -pon|--PoN Panel of Normals - default:refs/pon.vcf.gz

Example usage:
"<path to SomaticExtract>/SomaticExtract.sh -n normal-LUAD-09.bam -t tumor-LUAD-09.bam -p pre-LUAD-09.bam -o LUAD09" 
