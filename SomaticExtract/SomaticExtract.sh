#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo "SCRIPT_DIR		= $SCRIPT_DIR"

mutect=${SCRIPT_DIR}/src/mutect.sh
parser=${SCRIPT_DIR}/src/parser.sh
name=${SCRIPT_DIR}/src/name.sh
mutectParser=${SCRIPT_DIR}/src/mutectParser.py
pon=${SCRIPT_DIR}/refs/pon.vcf.gz

POSITIONAL=()
while [[ $# -gt 0 ]]; do
	key="$1"

  	case $key in
    	-n|--normal)
			NORMAL="$2"
			shift # past argument
			;;
        -t|--tumor)
			TUMOR="$2"
			shift # past argument
			;;
        -p|--plasma)
			PLASMA="$2"
			shift # past argument
			;;
        -o|--OUT_PREFIX)
			NAME="$2"
			shift # past argument
			;;
        -g|--germline)
			germline="$2"
			shift # past argument
			;;
        -r|--REFERENCE)
			REF="$2"
			shift # past argument
			;;
		-pon|--PoN)
			PoN="$2"
			shift # past argument
			;;
		*)    # unknown option
			POSITIONAL+=("$1") # save it in an array for later
			shift # past argument
			;;
  	esac
done

if [[ $REF == "" ]]; then
	REF=/data/hadasvol/refs/HumanG1Kv37/human_g1k_v37.fasta
fi
if [[ $germline == "" ]]; then
	gnomad=/data/hadasvol/refs/HumanG1Kv37/af-only-gnomad.raw.sites.b37.vcf.gz
fi
if [[ $germline == "" ]]; then
	PoN=$pon	
fi

set -- "${POSITIONAL[@]}" # restore positional parameters

OUT=results_${NAME}
echo "OUT_PREFIX		= ${NAME}"
echo "NORMAL			= ${NORMAL}"
echo "TUMOR			= ${TUMOR}"
echo "PLASMA                = ${PLASMA}"
echo "REF			= ${REF}"
echo "OUT			= ${OUT}"
echo

mkdir -p results_${NAME}/vcf

echo "Get normal sample name and index bams"
echo "sbatch $name $NORMAL $OUT $NAME $TUMOR $PLASMA"
jobID1=$(sbatch $name $NORMAL $OUT $NAME $TUMOR $PLASMA)
jobArray1=($(echo $jobID1 | tr ' ' ' '))
jobID1=${jobArray1[3]}
echo

echo "Mutect2"
echo "sbatch --dependency=aftercorr:$jobID1 $mutect $REF $TUMOR $NORMAL $gnomad $OUT $NAME $PoN"
jobID=$(sbatch --dependency=aftercorr:$jobID1 $mutect $REF $TUMOR $NORMAL $gnomad $OUT $NAME $PoN )
jobArray=($(echo $jobID | tr ' ' ' '))
jobID=${jobArray[3]}
echo

echo "Parse filtered vcf and output somatic reads in plasma"
echo "sbatch --dependency=aftercorr:$jobID $parser $OUT $NAME $REF $mutectParser $PLASMA"
sbatch --dependency=aftercorr:$jobID $parser $OUT $NAME $REF $mutectParser $PLASMA
