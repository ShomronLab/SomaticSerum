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
		
		self.read_names = set()
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
		
		def _cover(x):
			A = 0
			C = 0
			G = 0
			T = 0
			for pileupcolumn in pre.pileup(x.CHROM, x.POS, x.POS+1, ignore_overlaps=False, min_base_quality=0):
					if pileupcolumn.pos < x.POS or pileupcolumn.pos > x.POS: continue
					for pileupread in pileupcolumn.pileups:
						if not pileupread.is_del and not pileupread.is_refskip:
							n = pileupread.alignment.query_sequence[pileupread.query_position]
							if n == 'A':
								A = A + 1
							elif n == 'C':
								C = C + 1
							elif n == 'G':
								G = G + 1
							elif n == 'T':
								T = T + 1
			return [A, C, G, T]
		
		def _match(x):
			for n in ['AU', 'CU', 'GU', 'TU']:
				if n[0] == x.ALT:
					if x['tumor ' + n] != 0 and x['pre ' + n] != 0:
						return 'x'
		
		def _match2(x):
			count_somatic = 0
			count_normal = 0
			if len(x.ALT) != 1 or len(x.REF) != 1:
				for pileupcolumn in pre.pileup(x.CHROM, x.POS, x.POS+1, ignore_overlaps=False, min_base_quality=0):
					if pileupcolumn.pos < x.POS or pileupcolumn.pos > x.POS: continue
					for pileupread in pileupcolumn.pileups:
						if pileupread.is_del:
							count_somatic = count_somatic + 1
							indel = 'del'
							self.read_names.add(pileupread.alignment.query_name)
							self.del_reads.add(pileupread.alignment.query_name)
							continue
						if pileupread.indel:
							count_somatic = count_somatic + 1
							indel = 'insert'
							self.read_names.add(pileupread.alignment.query_name)
							self.ins_reads.add(pileupread.alignment.query_name)
							continue
						self.normal_reads.add(pileupread.alignment.query_name)
						count_normal = count_normal + 1
				if count_somatic == 0:
					return [np.nan, count_normal, count_somatic]
				return ['{} {}'.format(indel, count_somatic), count_normal, count_somatic]
			
			for n in ['A', 'C', 'G', 'T']:
				if n == x.ALT and x['pre {}'.format(n)] != 0:
					for pileupcolumn in pre.pileup(x.CHROM, x.POS, x.POS+1, ignore_overlaps=False, min_base_quality=0):
						if pileupcolumn.pos < x.POS or pileupcolumn.pos > x.POS: continue
						for pileupread in pileupcolumn.pileups:
							try:
								if pileupread.alignment.query_sequence[pileupread.query_position] == x.ALT:
									self.read_names.add(pileupread.alignment.query_name)
									self.snp_reads.add(pileupread.alignment.query_name)
									count_somatic = count_somatic + 1
								else:
									self.normal_reads.add(pileupread.alignment.query_name)
									count_normal = count_normal + 1
							except TypeError:
								print(pileupread.alignment.query_sequence, pileupread.query_position, x.ALT)
					return ['x', count_normal, count_somatic]
			return [np.nan, count_normal, count_somatic]	
		
        def _cover2(x):
            A = C = G = T = d = i = []
            mapping = {'A':A, 'C':C, 'G':G, 'T':T}
            for pileupColumn in pre.pileup(x.CHROM, x.POS-1, x.POS, stepper='samtools', min_base_quality=0, ignore_overlaps=False, ignore_orphans=False, truncate=True):
                for pileupRead in pileupColumn.pileups:
                    if pileupRead.indel > 0:
                        i.append(pileupRead.alignment.query_name)
                    elif pileupRead.indel < 0:
                        d.append(pileupRead.alignment.query_name)
                    
                    n = pileupRead.alignment.query_sequence[pileupRead.query_position]
                    mapping[n].append(pileupRead.alignment.query_name)

            sum_snvs = len(mapping['A']) + len(mapping['C']) + len(mapping['G']) + len(mapping['T'])
            mappings = [mapping['A'], mapping['C'], mapping['G'], mapping['T']]
            if len(x.ALT) > len(x.REF):
                count_somatic = len(i)
                count_normal = len(d) + sum_snvs
                match = 'ins' if count_somatic > 0 else np.nan
                return mappings.extend(match, count_normal, count_somatic)
            
            if len(x.ALT) < len(x.REF):
                count_somatic = len(d)
                count_normal = len(i) + sum_snvs
                match = 'del' if count_somatic > 0 else np.nan
                return mappings.extend(match, count_normal, count_somatic)
            
            def __snvreturn(n):
                count_somatic = len(mapping[n])
                count_normal = len(i) + len(d) + sum_snvs - count_somatic
                match = 'snv' if count_somatic > 0 else np.nan
                return mappings.extend(match, count_normal, count_somatic)

            if x.ALT == 'A':
                __snvreturn('A')
            elif x.ALT == 'C':
                __snvreturn('C')
            elif x.ALT == 'G':
                __snvreturn('G')
            elif x.ALT == 'T':
                __snvreturn('T')


		# df[['pre A','pre C','pre G','pre T']] = df.apply(lambda x : _cover(x), axis=1, result_type ='expand')
		if self.Indel == 'indel':
			df['match'] = df.apply(lambda x: _match(x), axis=1)
		# if self.Indel == 'mutect':
		# 	df[['match', 'Normal Count', 'Somatic Count']] = df.apply(lambda x: _match2(x), axis=1, result_type ='expand')
        if self.Indel == 'mutect':
			df[['pre A','pre C','pre G','pre T', 'match', 'Normal Count', 'Somatic Count']] = df.apply(lambda x: _cover2(x), axis=1, result_type ='expand')
		
		df.to_pickle(self.pickle_file)
		df.to_csv(self.csv_file, index=False)
		
		self.vcfdf = df

		return df
	

	def writeOut(self) -> None:
		with open(self.somatic_reads_file, 'a') as f:
			f.writelines("%s\n" % l for l in self.read_names)
		with open(self.normal_reads_file, 'a') as f:
			f.writelines("%s\n" % l for l in self.normal_reads)


	def extract_reads(self) -> None:
		if self.pickle:
			n = [x.rstrip() for x in self.somatic_reads]
			print(n)
		else:
			n = list(self.read_names)
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