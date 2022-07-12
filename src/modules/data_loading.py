import logging
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import torch
from config import DATA_DIR
from utils.utils import debug


@dataclass
class BamRead:
    bam_file: Path
    read_num: int
    label: int
    seq: list
    qual: list
    cigar: list
    meta: list


class DataLoader:
    READ_LENGTH = 200
    LABELS_DICT = {
        'somatic': 1,
        'normal': 0
    }

    @staticmethod
    def __get_cached_file_name(bam_dir):
        return '__'.join(str(bam_dir).split('/'))

    @ classmethod
    def _get_chached_file_path(cls, bam_dir):
        cached_name = cls.__get_cached_file_name(bam_dir)
        return DATA_DIR / f"{cached_name}.pkl"

    @ classmethod
    def load_from_dir(cls, bam_dir, force=False):
        if not type(bam_dir) == Path:
            bam_dir = Path(bam_dir)

        chached_file_path = cls._get_chached_file_path(bam_dir)

        if os.path.exists(chached_file_path) and force:
            os.remove(chached_file_path)

        if os.path.exists(chached_file_path):
            debug("Loading from cache")
            with open(chached_file_path) as chached_file:
                number_of_normal_reads, number_of_somatic_reads, seqs = pickle.load(
                    chached_file)

        bam_files = cls.get_bam_files(bam_dir)
        data = cls.read_bam_files(bam_files)
        return data

    @ classmethod
    def get_bam_files(cls, bam_dir):
        bam_files = []

        for root, dirs, files in os.walk(bam_dir):
            for file in files:
                if file.endswith(".bam"):
                    bam_files.append(bam_dir / file)
        return bam_files

    @ classmethod
    def label_from_bam_file(cls, bam_file):
        labels_name = bam_file.stem.split("_")[0]
        return cls.LABELS_DICT[labels_name]

    @ classmethod
    def longest(cls, bam_files):
        """
        Returns the longest read in the bam file.
        """
        longest_read = 0
        for bam_file in bam_files:
            bam = pysam.AlignmentFile(bam_file, 'rb')
            for read in bam.fetch():
                if read.template_length > 1000:
                    debug(
                        f'template length ({read.template_length}) is larger than 1000')
                    continue
                if read.template_length > longest_read:
                    longest_read = read.template_length
            bam.close()
        debug("Longest read: {}".format(longest_read))
        return longest_read

    @ classmethod
    def data_from_read(cls, read, longest_read):
        read_dict = read.to_dict()
        match = re.findall(r'\D+|\d+', read_dict['cigar'])
        seqs_lengths = [int(x) for x in match[::2]]
        cigars = list(match[1::2])

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
                seq = seq  # [nuc_float[x] for x in seq]
                quals = [round((ord(x)-33)/93, 3)
                         for x in read_dict['qual'][k:seq_len+k]]
                cc = [cigar] * seq_len  # [cigar_dict[cigar]] * seq_len
                k += seq_len
            seqs_full.extend(seq)
            quals_full.extend(quals)
            cigars_full.extend(cc)

        tlen = 0 if abs(read.template_length) > 1000 else abs(
            read.template_length)
        # print(read.get_tags())
        meta = {
            'tlen': tlen,
            'query_length': read.query_length,
            'tlen_normalized': round(tlen/longest_read, 3),
            'is_reverse': int(read.is_reverse),
            'is_read1': int(read.is_read1),
            'is_read2': int(read.is_read2),
            'mapping_quality': round(read.mapping_quality/60, 3),
            'query_length_normalized': round(read.query_length/longest_read, 3),
            'tag_NM': round(read.get_tag('NM')/100, 3),
            'tag_AS': round(read.get_tag('AS')/500, 3),
            'tag_XS': round(read.get_tag('XS'), 3),
            'tag_MC': float(re.match('\d*', read.get_tag('MC'))
                            [0]) if 'MC' in dict(read.get_tags()).keys() else -1,
        }

        seqs_full += [-1] * (cls.READ_LENGTH - len(seqs_full))
        quals_full += [-1] * (cls.READ_LENGTH - len(quals_full))
        cigars_full += [-1] * (cls.READ_LENGTH - len(cigars_full))
        # meta += [-1] * (self.READ_LENGTH - len(meta))

        seqs_full = seqs_full[:cls.READ_LENGTH]
        quals_full = quals_full[:cls.READ_LENGTH]
        cigars_full = cigars_full[:cls.READ_LENGTH]
        # meta = meta[:self.READ_LENGTH]

        return {
            'seq': seqs_full,
            'qual': quals_full,
            'cigar': cigars_full,
            'meta': meta
        }

    @ classmethod
    def read_bam_files(cls, bam_files):
        seqs = []
        max_read_length = cls.longest(bam_files)
        for i, bam_file in enumerate(bam_files):
            bam = pysam.AlignmentFile(bam_file, 'rb')
            debug(f"Loading reads from: {bam_file} \t\tbam file number {i}")
            label = cls.label_from_bam_file(bam_file)
            for read_num, read in enumerate(bam.fetch()):
                if read.is_secondary or read.is_supplementary:
                    continue
                data = cls.data_from_read(read, max_read_length)
                seqs.append(
                    BamRead(
                        bam_file=bam_file,
                        read_num=read_num,
                        label=label,
                        **data
                    )
                )
            bam.close()
        return seqs
