import io
import os
import pandas as pd
import numpy as np
import gzip
import sys
import pysam


info_snp = ['SOMATIC','QSS','TQSS','NT','QSS_NT','TQSS_NT','SGT','DP','MQ','MQ0','ReadPosRankSum','SNVSB','SomaticEVS']
info_indel = ['SOMATIC','QSI','TQSI','NT','QSI_NT','TQSI_NT','SGT','MQ','MQ0','RU','RC','IC','IHP','SomaticEVS', 'empty']
info_mutect = ['AS_FilterStatus','AS_SB_TABLE','DP','ECNT','GERMQ','MBQ','MFRL','MMQ','MPOS','NALOD','NLOD','POPAF','RPA','RU','STR','STRQ','TLOD']

form_snp = ['DP','FDP','SDP','SUBDP','AU','CU','GU','TU']
form_indel = ['DP','DP2','TAR','TIR','TOR','DP50','FDP50','SUBDP50','BCN50']
form_mutect = ['GT','AD','AF','DP','F1R2','F2R1','FAD','SB']

columns_snp = ['CHROM','POS','REF','ALT','FORMAT','NORMAL','TUMOR','QSS','DP','MQ','MQ0','ReadPosRankSum','SomaticEVS']
columns_indel = ['CHROM','POS','REF','ALT','FORMAT','NORMAL','TUMOR','QSI','MQ','MQ0','SGT','MQ0','RU','RC','IC','IHP','SomaticEVS']
columns_mutect = ['CHROM','POS','REF','ALT','FORMAT','NORMAL','TUMOR','GT','AD','AF','DP','F1R2','F2R1','FAD','SB']

chrs = [str(i) for i in range(1,23)]
chrs.extend(['X', 'Y', 'MT'])


class VcfParser:
	def __init__(self, vcf_path, bam_path, indel, name, out) -> None:
		self.vcf_path = vcf_path
		self.bam_path = bam_path
		self.name = name
		self.out = out
		self.Indel = indel
		self.csv_file = '{}/{}.somatic.{}.parse.csv'.format(self.out, self.name, self.Indel)
		self.pickle_file = '{}/{}.somatic.{}.parse.pkl'.format(self.out, self.name, self.Indel)
		self.somatic_reads_file = '{}/somatic_reads.{}.txt'.format(self.out, self.name)
		self.normal_reads_file = '{}/normal_reads.{}.txt'.format(self.out, self.name)

		if self.Indel == 'indel':
			self.Info = info_indel
			self.Form = form_indel
			self.Columns = columns_indel
		elif self.Indel == 'mutect':
			self.Info = info_mutect
			self.Form = form_mutect
			self.Columns = columns_mutect
		else:
			self.Info = info_snp
			self.Form = form_snp
			self.Columns = columns_snp
		
		self.pickle = False
		if os.path.isfile(self.pickle_file):
			self.vcfdf = pd.read_pickle(self.pickle_file)
			self.pickle = True
			with open(self.somatic_reads_file) as f:
				self.somatic_reads = f.readlines()
		else:
			with gzip.open(self.vcf_path, 'rt') as f:
				lines = [l for l in f if not l.startswith('##')]
				self.vcfdf = pd.read_csv(io.StringIO(''.join(lines)), 
								dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': str, 'FILTER': str, 'INFO': str},
								sep='\t'
								).rename(columns={'#CHROM': 'CHROM'})
		
		self.somatic_reads = set()
		self.snp_reads = set()
		self.del_reads = set()
		self.ins_reads = set()
		self.normal_reads = set()
			
			
	def parse_df(self) -> pd.DataFrame:
		if self.pickle: return self.vcfdf

		df = self.vcfdf[(self.vcfdf.FILTER == 'PASS') & (self.vcfdf.CHROM.isin(chrs))]
		print(self.vcfdf)
		# if self.Indel == 'mutect': return df
		df[self.Info] = df['INFO'].str.split(';',expand=True)
		for col in self.Info:
			try:
				df[col] = df[col].str.split('=').str[-1]
			except ValueError:
				df[col] = df[col].str.split()
		if self.Indel == 'mutect': return df
		tumor = ['tumor '+i for i in self.Form]
		df[tumor] = df['TUMOR'].str.split(':',expand=True)
		columns = self.Columns
		columns.extend(tumor)
		normal = ['normal '+i for i in self.Form]
		df[normal] = df['NORMAL'].str.split(':',expand=True)
		columns.extend(normal)
		for col in normal + tumor:
			df[col] = df[col].str.split(',').str[0]
		df = df[columns]
		df.drop(['FORMAT', 'NORMAL', 'TUMOR'],axis=1, inplace=True)
		if not self.Indel == 'mutect':
			df.sort_values(by='SomaticEVS', ascending=False, inplace=True)

		df.to_pickle(self.pickle_file)
		return df

	def precoverage(self, df) -> pd.DataFrame:
		# if self.pickle: return self.vcfdf
		pre = pysam.AlignmentFile(self.bam_path, 'rb')
		
		def _cover2(x):
			count_somatic, count_normal = 0, 0
			A, C, G, T, N, d, i = ([] for i in range(7))
			mapping = {'A':A, 'C':C, 'G':G, 'T':T, 'N':N}
			for pileupColumn in pre.pileup(x.CHROM, x.POS-1, x.POS, stepper='samtools', min_base_quality=0, ignore_overlaps=False, ignore_orphans=False, truncate=True):
				for pileupRead in pileupColumn.pileups:
					if pileupRead.query_position == None: continue
					if pileupRead.indel > 0:
						i.append(pileupRead.alignment.query_name)
					elif pileupRead.indel < 0:
						d.append(pileupRead.alignment.query_name)
					
					n = pileupRead.alignment.query_sequence[pileupRead.query_position]
					mapping[n].append(pileupRead.alignment.query_name)
			
			sum_snvs = len(mapping['A']) + len(mapping['C']) + len(mapping['G']) + len(mapping['T']) + len(mapping['N'])
			mappings = [len(mapping['A']), len(mapping['C']), len(mapping['G']), len(mapping['T']), len(mapping['N'])]
			reads = A + C + G + T + i + d
			
			def __snvreturn(n):
				count_somatic = len(mapping[n])
				count_normal = len(i) + len(d) + sum_snvs - count_somatic
				if count_somatic > 0:
					match = 'snv'
					self.somatic_reads.update(set(mapping[n]))
					normal = set(reads) - set(mapping[n])
					self.normal_reads.update(normal)
				else: 
					match = np.nan
				mappings.extend([match, count_normal, count_somatic])
				return mappings

			if len(x.ALT) > len(x.REF):
				count_somatic = len(i)
				count_normal = len(d) + sum_snvs
				if count_somatic > 0:
					match = 'ins'
					self.somatic_reads.update(set(i))
					self.normal_reads.update(set(reads) - set(i))  
				else: 
					match = np.nan
				mappings.extend([match, count_normal, count_somatic])
				return mappings
			
			if len(x.ALT) < len(x.REF):
				count_somatic = len(d)
				count_normal = len(i) + sum_snvs
				if count_somatic > 0:
					match = 'del'
					self.somatic_reads.update(set(d))
					self.normal_reads.update(set(reads) - set(d))
				else: 
					match = np.nan
				mappings.extend([match, count_normal, count_somatic])
				return mappings
			
			if x.ALT == 'A':
				return __snvreturn('A')
			elif x.ALT == 'C':
				return __snvreturn('C')
			elif x.ALT == 'G':
				return __snvreturn('G')
			elif x.ALT == 'T':
				return __snvreturn('T')
			
			mappings.extend([np.nan, 0, 0])
			return mappings
		
		
		if self.Indel == 'mutect':
			df[['pre A','pre C','pre G','pre T', 'pre N', 'match', 'Normal Count', 'Somatic Count']] = df.apply(lambda x: _cover2(x), axis=1, result_type ='expand')
		
		df.to_pickle(self.pickle_file)
		df.to_csv(self.csv_file, index=False)
		
		self.vcfdf = df
		pre.close()
		return df
	

	def writeOut(self) -> None:
		with open(self.somatic_reads_file, 'a') as f:
			f.writelines("%s\n" % l for l in self.somatic_reads)
		with open(self.normal_reads_file, 'a') as f:
			f.writelines("%s\n" % l for l in self.normal_reads)


	def extract_reads(self) -> None:
		if self.pickle:
			n = [x.rstrip() for x in self.somatic_reads]
			print(n)
		else:
			n = list(self.somatic_reads)
		bamfile = pysam.AlignmentFile(self.bam_path, 'rb')
		name_indexed = pysam.IndexedReads(bamfile)
		name_indexed.build()
		header = bamfile.header.copy()
		out = pysam.Samfile('{}/{}.{}.bam'.format(self.out, 'somatic_reads', self.name), 'wb', header=header)
		for name in n:
			print(name)
			try:
				name_indexed.find(name)
			except KeyError:
				pass
			else:
				iterator = name_indexed.find(name)
				for x in iterator:
					out.write(x)


if __name__ == "__main__":
	# Arguments: vcf_path, bam_path, indel, name, out
	data = VcfParser(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
	vcfdf = data.precoverage(data.parse_df())
	data.writeOut()
	# data.extract_reads()

	print(vcfdf)
	# print("snp reads")
	# print(data.snp_reads)
	# print("del read")
	# print(data.del_reads)
	# print("insert reads")
	# print(data.ins_reads)