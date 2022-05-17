import numpy as np
import pysam
import re
import csv
import random
import os, glob
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pysam

# naming of sample files should be in the format of: <TYPE>_reads.<SAMPLENAME>.bam like "normal_reads.LUAD34.bam"

class CustomBamDataset(Dataset):
    def __init__(self, bam_file_location, out, channels, w_set, bam_lines_maximum = 9999999999, read_length = 151):
        self.bam_dir = bam_file_location
        self.out = out
        self.channels = channels
        self.bam_lines_maximum = bam_lines_maximum
        self.read_length = read_length
        self.bam_files = []
        self.bam_file_labels = {}
        self.bam_file_sample_names = {}
        self.number_of_somatic_reads = 0
        self.number_of_normal_reads = 0
        self.max_tlen = 1
        self.reads = []
        for root, dirs, files in os.walk(self.bam_dir):
            for file in files:
                if file.endswith(".bam"):
                    self.bam_files.append(file)
                    self.bam_file_labels[file] = file.split("_")[0]
                    self.bam_file_sample_names[file] = file.split(".")[1]

        if not os.path.exists("{}/bam_data_{}.npy".format(self.out, w_set)) and not os.path.exists("{}/data_{}.p".format(self.out, w_set)):
            # Loading reads into memory - Add here filters based on samples/variants/etc...
            print("\nloading bam files")
            for bam_number, bam_file in enumerate(self.bam_files):
                print(bam_number, end=" ")

                bam = pysam.AlignmentFile(self.bam_dir+bam_file, 'rb')
                curr_file_read_list = []
                for read_pair_number, (read1, read2) in enumerate(self.read_pair_generator(bam)):
                    if read1 == None or read2 == None: continue
                    curr_file_read_list.append([bam_file, read1.query_sequence, read1.reference_id,
                                                read1.query_alignment_start, read1.template_length, read1.is_reverse, read2.query_sequence, 
                                                read1.query_qualities, read2.query_qualities,
                                                read1.query_length, read2.query_length,
                                                read1.mapping_quality, read2.mapping_quality ])
                    if read1.template_length > self.max_tlen: self.max_tlen = read1.template_length
                if self.bam_file_labels[bam_file] == "somatic":
                    self.number_of_somatic_reads += min(self.bam_lines_maximum, len(curr_file_read_list) )
                else:
                    self.number_of_normal_reads += min(self.bam_lines_maximum, len(curr_file_read_list) )
                self.reads = self.reads + random.sample(curr_file_read_list, min(self.bam_lines_maximum, len(curr_file_read_list) ))
                bam.close()

            self.reads_np = np.array(self.reads, dtype=object)
            np.save("{}/bam_data_{}.npy".format(self.out, w_set), self.reads_np)
            with open("{}/data_{}.p".format(self.out, w_set), 'wb') as f:
                pickle.dump((self.number_of_normal_reads, self.number_of_somatic_reads, self.bam_file_labels), f)
        else:
            print("loading data from pickle") 
            self.reads_np = np.load("{}/bam_data_{}.npy".format(self.out, w_set), allow_pickle=True)
            with open("{}/data_{}.p".format(self.out, w_set), 'rb') as f:
                self.number_of_normal_reads, self.number_of_somatic_reads, self.bam_file_labels = pickle.load(f)
                
        print("\nfinished loading files for {}".format(w_set))
        print("number of total read-pairs: ", len(self.reads_np))
        print("number of somatic/normal read-pairs: {} / {}, {}".format(self.number_of_somatic_reads, self.number_of_normal_reads, self.number_of_somatic_reads/self.number_of_normal_reads))


    def __len__(self):
        return len(self.reads_np)


    def __getitem__(self, idx):
        # read_pair format:
        # 0 - bam file name
        # 1 - read1 sequence
        # 2 - read1 reference id
        # 3 - read1 alignment start
        # 4 - read1 template length
        # 5 - read1 aligned in reverse
        # 6 - read2 sequence
        read_pair = self.reads_np[idx]
        read1_seq = read_pair[1]
        read2_seq = read_pair[6]
        tlen = read_pair[4]
        read1_qual = read_pair[7]
        read2_qual = read_pair[8]
        read1_len = read_pair[9]
        read2_len = read_pair[10]
        read1_mq = read_pair[11]
        read2_mq = read_pair[12]

        label_string = self.bam_file_labels[read_pair[0]]
        if label_string == "somatic":
            label = torch.tensor(1, dtype=torch.long)
        elif label_string == "normal":
            label = torch.tensor(0, dtype=torch.long)

        ## translating the ATGCN sequence into a tensor on 12345
        ## also, combines the two reads, with maximum length of 151*2 and have a [-1] values if the reads are shorter
        combined_read = torch.negative(torch.ones(self.channels, int(self.read_length*2), dtype=torch.float))
        for i, nuc in enumerate(read1_seq):
            if nuc == "A":
                combined_read[0][i] = 0.2
            elif nuc == "T":
                combined_read[0][i] = 0.4
            elif nuc == "G":
                combined_read[0][i] = 0.6
            elif nuc == "C":
                combined_read[0][i] = 0.8
            elif nuc == "N":
                combined_read[0][i] = 1.0

        for i, nuc in enumerate(reversed(read2_seq)):
            if nuc == "A":
                combined_read[0][-i-1] = 0.2
            elif nuc == "T":
                combined_read[0][-i-1] = 0.4
            elif nuc == "G":
                combined_read[0][-i-1] = 0.6
            elif nuc == "C":
                combined_read[0][-i-1] = 0.8
            elif nuc == "N":
                combined_read[0][-i-1] = 1.0
            
        for i, q in enumerate(read1_qual):
            combined_read[1][i] = q/93
        for i, q in enumerate(reversed(read2_qual)):
            combined_read[1][-i-1] = q/93
        
        combined_read[2][0] = tlen/self.max_tlen
        combined_read[2][1] = read1_len/151
        combined_read[2][2] = read2_len/151
        combined_read[2][3] = read1_mq/93

        return combined_read, label


    def read_pair_generator(self, bam, region_string=None):
        """
        Generate read pairs in a BAM file or within a region string.
        Reads are added to read_dict until a pair is found.
        """
        read_dict = defaultdict(lambda: [None, None])
        for read in bam.fetch():
            if not read.is_proper_pair or read.is_secondary or read.is_supplementary:
                continue
            qname = read.query_name
            if qname not in read_dict:
                if read.is_read1:
                    read_dict[qname][0] = read
                else:
                    read_dict[qname][1] = read
            else:
                if read.is_read1:
                    yield read, read_dict[qname][1]
                else:
                    yield read_dict[qname][0], read
                del read_dict[qname]