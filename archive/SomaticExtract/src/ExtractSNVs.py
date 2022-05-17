import sys, os, logging
import random
import pysam
import pandas as pd

vcf_file = sys.argv[1]
pre_file = sys.argv[2]
luad = sys.argv[3]
out = sys.argv[4]
rand = sys.argv[5]

logging.basicConfig(filename = "{}/ExtractReads.{}.log".format(out,luad),
                    filemode = "a",
                    format = "%(levelname)s %(asctime)s - %(message)s", 
                    level = logging.DEBUG)
logger = logging.getLogger()
logger.info("Started ExtractReads.py")

somatic_file = "{}/somatic_reads.{}.SNVs.bam".format(out, luad)
normal_file = "{}/normal_reads.{}.SNVs.bam".format(out, luad)
somatic_srt_file = "{}/somatic_reads.{}.SNVs.srt.bam".format(out, luad)
normal_srt_file = "{}/normal_reads.{}.SNVs.srt.bam".format(out, luad)

bam = pysam.AlignmentFile(pre_file, "rb")
somatic = pysam.AlignmentFile(somatic_file, 'wb', template = bam)
normal = pysam.AlignmentFile(normal_file, 'wb', template = bam)
vcf = pysam.VariantFile(vcf_file,'r')

postions_in_plasma = 0
overall_postions = 0
somatic_reads = 0
normal_reads = 0
for rec in vcf.fetch():
    # print("{} {} {} {}".format(rec.chrom, rec.start, rec.ref, rec.alts[0]))
    for pile in bam.pileup(str(rec.chrom), rec.start, rec.stop, truncate=True):
        ref_reads = []
        ref_reads_names = []
        alt_reads = []
        alt_reads_names = []
        exclude_reads = []
        for r in pile.pileups:
            if r.query_position == None: continue
            # if len(rec.ref) < len(rec.alts[0]):
                
            nuc = r.alignment.query_sequence[r.query_position]
            if nuc == rec.ref:
                ref_reads.append(r)
                ref_reads_names.append(r.alignment.query_name)
            elif nuc == rec.alts[0]:
                alt_reads.append(r)
                alt_reads_names.append(r.alignment.query_name)
        for name in ref_reads_names:
            if name in alt_reads_names:
                exclude_reads.append(name)
                logger.info("read {} excluded".format(name))
                logger.info("{} {} {} {}".format(rec.chrom, rec.start, rec.ref, rec.alts[0]))
                # print("read {} excluded".format(name))
        if len(alt_reads) > 0:
            postions_in_plasma += 1
            for alt in alt_reads:
                if alt.alignment.query_name in exclude_reads: continue
                somatic.write(alt.alignment)
                try:
                    somatic.write(bam.mate(alt.alignment))
                except ValueError:
                    pass
            if rand == 'True':
                try:
                    ref_to_write = random.sample(ref_reads, len(alt_reads))
                except:
                    ref_to_write = ref_reads
            else:
                ref_to_write = ref_reads
            for ref in ref_to_write:
                if ref.alignment.query_name in exclude_reads: continue
                normal.write(ref.alignment)
                try:
                    normal.write(bam.mate(ref.alignment))
                except ValueError:
                    pass
            somatic_reads += len(alt_reads)
            normal_reads += len(ref_to_write)
    overall_postions += 1
    # sys.exit()

try:
    pos_ratio = postions_in_plasma/overall_postions
except:
    pos_ratio = 0

logger.info("# Postions with alt: {}, {}".format(postions_in_plasma, pos_ratio))
logger.info("# Somatic reads: {}".format(somatic_reads))
logger.info("# Normal reads: {}".format(normal_reads))

bam.close()
vcf.close()
somatic.close()
normal.close()

pysam.sort("-o", somatic_srt_file, somatic_file)
pysam.index(somatic_srt_file)
if os.path.exists(somatic_file): os.remove(somatic_file)

pysam.sort("-o", normal_srt_file, normal_file)
pysam.index(normal_srt_file)
if os.path.exists(normal_file): os.remove(normal_file)

