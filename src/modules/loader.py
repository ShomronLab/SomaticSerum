import logging
import os
import pickle
import re
import sys

import numpy as np
import pysam
import torch


class CustomBamDataset2(torch.utils.data.Dataset):

    nuc_float = {'A': 0.0, 'C': 0.2, 'G': 0.4, 'T': 0.6, 'N': 0.8, None: 1.0}
    cigar_pat = re.compile(r"\d+[MIDNSHP=X]{1}")
    cigar_dict = {"M": 0.1,  # M 0 alignment match (can be a sequence match or mismatch)
                  "I": 0.2,  # I 1 insertion to the reference
                  "D": 0.3,  # D 2 deletion from the reference
                  "N": 0.4,  # N 3 skipped region from the reference
                  # S 4 soft clipping (clipped sequences present in SEQ)
                  "S": 0.5,
                  # H 5 hard clipping (clipped sequences NOT present in SEQ)
                  "H": 0.6,
                  # P 6 padding (silent deletion from padded reference)
                  "P": 0.7,
                  "=": 0.8,  # = 7 sequence match
                  "X": 0.9   # X 8 sequence mismatch
                  }

    def __init__(self, bam_file_location, out=None, whichSet="test_loader", read_length=200, input_files_path=None, force=False):
        self.READ_LENGTH = read_length
        self.out = out
        self.pickles = '/'.join(self.out.split("/")[:-1])
        self.whichSet = whichSet
        self.bam_dir = bam_file_location

        self.bam_files = []
        self.bam_file_labels = {}
        self.bam_file_sample_names = {}
        self.number_of_somatic_reads = 0
        self.number_of_normal_reads = 0

        chached_file_path = self._get_chached_file_path()

        logging.info("Dataloading {}".format(self.whichSet))
        if os.path.exists(chached_file_path) and force:
            os.remove(chached_file_path)

        if os.path.exists(chached_file_path):
            print(f"File {chached_file_path} exists - using as {self.whichSet}")
            with open(chached_file_path) as chached_file:
                self.number_of_normal_reads, self.number_of_somatic_reads, self.seqs = pickle.load(
                    chached_file)
        else:
            if isinstance(self.bam_dir, list):
                for file in self.bam_dir:
                    self.bam_dir = input_files_path
                    self.out = os.path.join(self.bam_dir, 'output')
                    file = file.split('/')[-1]
                    if file.endswith(".bam"):
                        self.bam_files.append(file)
            else:
                for root, dirs, files in os.walk(self.bam_dir):
                    for file in files:
                        if file.endswith(".bam"):
                            self.bam_files.append(file)
            for file in self.bam_files:
                self.bam_file_labels[file] = file.split("_")[0]
                self.bam_file_sample_names[file] = file.split(".")[1]
            self.seqs = []
            self.genSet()
            with open(self._get_chached_file_path(), 'wb') as f:
                pickle.dump((self.number_of_normal_reads,
                            self.number_of_somatic_reads, self.seqs), f)

        logging.info("Number of Normal reads: {}, Number of Somatic reads: {}. Ratio: {}".format(
            self.number_of_normal_reads,
            self.number_of_somatic_reads,
            round(self.number_of_normal_reads/self.number_of_somatic_reads, 3)))

    def _get_chached_file_path(self):
        return "{}/ {}.pkl".format(self.pickles, self.whichSet)

    def __len__(self):
        return len(self.seqs)

    def longest(self):
        """
        Returns the longest read in the bam file.
        """
        longest_read = 0
        for bam_number, bam_file in enumerate(self.bam_files):
            bam = pysam.AlignmentFile(
                os.path.join(self.bam_dir, bam_file), 'rb')
            for read in bam.fetch():
                if read.template_length > 1000:
                    continue
                if read.template_length > longest_read:
                    longest_read = read.template_length
            bam.close()
        logging.info("Longest read: {}".format(longest_read))
        return longest_read

    def genSet(self):
        self.longest_read = self.longest()
        for bam_number, bam_file in enumerate(self.bam_files):
            bam = pysam.AlignmentFile(
                os.path.join(self.bam_dir, bam_file), 'rb')
            logging.info("Loading reads from: {} \t\tbam file number {}".format(
                bam_file, bam_number))
            for read in bam.fetch():
                if read.is_secondary or read.is_supplementary:
                    continue

                def genertor(read):
                    read_dict = read.to_dict()
                    match = re.findall(r'\D+|\d+', read_dict['cigar'])
                    seqs_lengths = [int(x) for x in match[::2]]
                    cigars = [x for x in match[1::2]]

                    k = 0
                    seqs_full = []
                    cigars_full = []
                    quals_full = []
                    for cigar, seq_len in zip(cigars, seqs_lengths):
                        if cigar in ['H', 'D', 'P']:
                            seq = [-1] * seq_len
                            quals = [-1] * seq_len
                            cc = [-1] * seq_len
                        else:
                            seq = []
                            seq[:0] = read_dict['seq'][k:seq_len+k]
                            seq = [CustomBamDataset2.nuc_float[x] for x in seq]
                            quals = [round((ord(x)-33)/93, 3)
                                     for x in read_dict['qual'][k:seq_len+k]]
                            cc = [CustomBamDataset2.cigar_dict[cigar]] * seq_len
                            k += seq_len
                        seqs_full.extend(seq)
                        quals_full.extend(quals)
                        cigars_full.extend(cc)

                    tlen = 0 if abs(read.template_length) > 1000 else abs(
                        read.template_length)
                    # print(read.get_tags())
                    meta = [
                        tlen,
                        read.query_length,
                        round(tlen/self.longest_read, 3),
                        int(read.is_reverse),
                        int(read.is_read1),
                        int(read.is_read2),
                        round(read.mapping_quality/60, 3),
                        round(read.query_length/self.longest_read, 3),
                        round(read.get_tag('NM')/100, 3),
                        round(read.get_tag('AS')/500, 3),
                        round(read.get_tag('XS'), 3),
                        float(re.match('\d*', read.get_tag('MC'))
                              [0]) if 'MC' in dict(read.get_tags()).keys() else -1,
                    ]

                    seqs_full += [-1] * (self.READ_LENGTH - len(seqs_full))
                    quals_full += [-1] * (self.READ_LENGTH - len(quals_full))
                    cigars_full += [-1] * (self.READ_LENGTH - len(cigars_full))
                    # meta += [-1] * (self.READ_LENGTH - len(meta))

                    seqs_full = seqs_full[:self.READ_LENGTH]
                    quals_full = quals_full[:self.READ_LENGTH]
                    cigars_full = cigars_full[:self.READ_LENGTH]
                    # meta = meta[:self.READ_LENGTH]

                    if self.bam_file_labels[bam_file] == "normal":
                        self.number_of_normal_reads += 1
                        label = torch.tensor(0, dtype=torch.long)
                    elif self.bam_file_labels[bam_file] == "somatic":
                        self.number_of_somatic_reads += 1
                        label = torch.tensor(1, dtype=torch.long)

                    # logging.info("seqs_lengths", seqs_lengths,
                    #              sum(seqs_lengths))
                    # logging.info("cigars", cigars)
                    # logging.info("seqs_full", seqs_full, len(seqs_full))
                    # logging.info("quals_full", quals_full, len(quals_full))
                    # logging.info("cigars_full", cigars_full, len(cigars_full))
                    # logging.info("length seq", len(read_dict['seq']))
                    # logging.info(read_dict, '\n')
                    # logging.info(label)
                    # for k, v in read_dict.items():
                    #     print(k, v)
                    # print("Seqs: ", seqs_full, " \n")
                    # print("Quals: ", quals_full, " \n")
                    # print("Cigars: ", cigars_full, " \n")
                    # print("Meta: ", meta, " \n")
                    # print("Label: ", label, " \n")
                    # print("\n\n\n\n")
                    # sys.exit()

                    return [[seqs_full, quals_full, cigars_full], meta, label]

                self.seqs.append(genertor(read))
            bam.close()

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        # logging.info(torch.from_numpy(np.array(seq[0], dtype=np.float32)))
        # logging.info(torch.FloatTensor(seq[1]))
        # sys.exit()
        return torch.from_numpy(np.array(seq[0], dtype=np.float32)), torch.FloatTensor(seq[1]), seq[2]


if "__main__" == __name__:
    test = CustomBamDataset2(
        "/data/hadasvol/projects/cancer_plasma/seqmerge/DLbams")
