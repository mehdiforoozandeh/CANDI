"""
File to prepare inputs for CANDI inference

"""

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import json
import pysam
import shutil
import contextlib
import numpy as np

from pred import CANDIPredictor # external import

ASSAYS=[
    'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac',
    'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3',
    'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac',
    'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac',
    'chipseq-control'
    ]

sequencing_platforms = [
    'Illumina Genome Analyzer IIx', 'Illumina Genome Analyzer',
    'Illumina Genome Analyzer IIe', 'Illumina HiSeq 2000',
    'Illumina Genome Analyzer II', 'Illumina HiSeq 4000',
    'Illumina HiSeq 2500', 'Illumina Genome Analyzer I',
    'Illumina NextSeq 500'
    ]

run_types = ['single-ended', 'paired-ended']


# ------------------------------------------------------------------------
# slightly modiied BAM_TO_SIGNAL
# ------------------------------------------------------------------------
class BAM_TO_SIGNAL(object):

    def __init__(self, bam_file, chr_sizes_file, input_mdata, output_mdata, resolution=25):

        self.bam_file = bam_file
        self.output_dir = os.path.join("/",*self.bam_file.split('/')[:-1])
        self.chr_sizes_file = chr_sizes_file
        self.resolution = resolution
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')
        self.input_mdata = input_mdata
        self.output_mdata = output_mdata

        self.read_chr_sizes()

    def read_chr_sizes(self):
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        self.chr_sizes = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def initialize_empty_bins(self):
        return {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage_pysam(self):
        bins = self.initialize_empty_bins()

        total_mapped_reads = 0 
        bins_with_reads = 0 

        read_lens = [] 

        for chr in self.chr_sizes:
            for read in self.bam.fetch(chr):
                if read.is_unmapped:
                    continue
                total_mapped_reads += 1  
                read_lens.append(read.reference_length)

                start_bin = read.reference_start // self.resolution
                end_bin = read.reference_end // self.resolution
                for i in range(start_bin, end_bin + 1):
                    if bins[chr][i] == 0:  
                        bins_with_reads += 1  
                    bins[chr][i] += 1
        
        # Calculate coverage as the percentage of bins with at least one read
        total_bins = sum(len(b) for b in bins.values())  
        coverage = (bins_with_reads / total_bins) if total_bins > 0 else 0

        mean_read_len = np.mean(np.array(read_lens))

        return bins, total_mapped_reads, coverage, mean_read_len

    def save_signal_metadata(self, depth, mean_read_len):
        
        file_name = os.path.join(self.output_dir, "input_metadata.json")
        mdict = {
            "depth":depth,
            "sequencing_platform":self.input_mdata["sequencing_platform"],
            "mean_read_len":mean_read_len,
            "run_type":self.input_mdata["run_type"],
            }

        with open(file_name, 'w') as file:
            json.dump(mdict, file, indent=4)

        file_name = os.path.join(self.output_dir, "output_metadata.json")
        mdict = {# TODO: get proper output metadata
            "depth":depth,
            "sequencing_platform":self.output_mdata["sequencing_platform"],
            "mean_read_len":mean_read_len,
            "run_type":self.output_mdata["run_type"],
            }
        
        mdict.update(self.output_mdata)

        with open(file_name, 'w') as file:
            json.dump(mdict, file, indent=4)

    def save_signal(self, bins):

        for chr, data in bins.items():
            file_name = os.path.join(self.output_dir, f"{chr}.npz")
            np.savez_compressed(file_name, np.array(data))
    
    def full_preprocess(self):

        data, depth, _, mean_read_len = self.calculate_coverage_pysam()
        self.save_signal(data)
        self.save_signal_metadata(np.log2(depth), mean_read_len)

# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
def process_bam(bam_file, input_mdata, output_mdata):
    # Process BAM to signals using existing BAM_TO_SIGNAL

    bam_processor = BAM_TO_SIGNAL(
        bam_file=bam_file,
        chr_sizes_file="./data/inf_debug_hg38.chrom.sizes", #TODO: know where to get these sizes from
        input_mdata=input_mdata,
        output_mdata=output_mdata
    )
    bam_processor.full_preprocess()

    os.remove(bam_file)
    if os.path.exists(f"{bam_file}.bai"):
        os.remove(f"{bam_file}.bai")

def process_metadata(metadata):
    # TODO: make sure of these

    seq_platform = metadata["sequencing_platform"]
    try:
        index = sequencing_platforms.index(seq_platform) + 1
    except:
        index = 0
    metadata["sequencing_platform"] = index

    run_type = metadata["run_type"]
    try:
        index = run_types.index(run_type)
    except:
        index = -1
    metadata["run_type"] = index

def process_input_data(bios_path, temp_path):

    os.makedirs(temp_path, exist_ok=True)

    exps = os.listdir(bios_path)

    for exp in exps:

        assert exp in ASSAYS

        exp_path = os.path.join(bios_path, exp)
        temp_exp_path = os.path.join(temp_path, exp)
        os.makedirs(temp_exp_path, exist_ok=True)

        input_mdata_path = os.path.join(exp_path, "input_metadata.json")
        try:
            with open(input_mdata_path) as f:
                input_mdata = json.load(f)
        except:
            input_mdata = {
                "sequencing_platform" : "N/A",
                "run_type" : "paired" # TODO: make this correct
            }
        process_metadata(input_mdata)

        output_mdata_path = os.path.join(exp_path, "output_metadata.json")
        try:
            with open(output_mdata_path) as f:  
                output_mdata = json.load(f)
        except:
            output_mdata = {
                "sequencing_platform" : "N/A",
                "run_type" : "N/A"
            }
        process_metadata(output_mdata)

        for file in os.listdir(exp_path):

            if file.endswith(".bam"):

                file_path = os.path.join(exp_path,file)
                temp_file_path = os.path.join(temp_exp_path,file)
                shutil.copy2(file_path, temp_file_path)
                pysam.index(temp_file_path)

                process_bam(temp_file_path, input_mdata, output_mdata)

# ------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------

def load_candi_predictor(model_path):

    with contextlib.redirect_stdout(None):
        model = CANDIPredictor(model_path)

    return model

# ------------------------------------------------------------------------
# Extra stuff
# ------------------------------------------------------------------------

def main():
    pass

if __name__=="__main__":
    main()
