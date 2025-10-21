"""
File to prepare inputs for CANDI inference

"""

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import pysam
import shutil
import contextlib
from data_utils import BAM_TO_SIGNAL
from model import CANDI, CANDI_UNET

# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
def process_bam(bam_file):
    # Process BAM to signals using existing BAM_TO_SIGNAL
    bam_processor = BAM_TO_SIGNAL(
        bam_file=bam_file,
        chr_sizes_file="./data/hg38.chrom.sizes"
    )
    bam_processor.full_preprocess(dsf_list=[1])

    os.remove(bam_file)
    if os.path.exists(f"{bam_file}.bai"):
        os.remove(f"{bam_file}.bai")

def iterate_over_data(base_dir, tmp_dir):

    bio_samples = os.listdir(base_dir)

    for bio_sample in bio_samples:

        bio_sample_path = os.path.join(base_dir, bio_sample)
        tmp_bio_sample_path = os.path.join(tmp_dir, bio_sample)
        os.makedirs(tmp_bio_sample_path, exist_ok=True)

        exps = os.listdir(bio_sample_path)

        for exp in exps:

            exp_path = os.path.join(bio_sample_path, exp)
            tmp_exp_path = os.path.join(tmp_bio_sample_path, exp)
            os.makedirs(tmp_exp_path, exist_ok=True)

            for file in os.listdir(exp_path):
                
                if file.endswith(".bam"):

                    file_path = os.path.join(exp_path,file)
                    tmp_file_path = os.path.join(tmp_exp_path,file)
                    shutil.copy2(file_path, tmp_file_path)
                    pysam.index(tmp_file_path)

                    with contextlib.redirect_stdout(None):
                        process_bam(tmp_file_path)

def process_input_data(args):

    data_path = args.data_path
    tmp_dir = args.tmp_path

    iterate_over_data(data_path, tmp_dir)

# ------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------

def make_inf_model(load_path):

    model = CANDI_UNET(
        signal_dim=25,
        metadata_embedding_dim=8,
        conv_kernel_size=3,
        n_cnn_layers=2,
        nhead=2,
        n_sab_layers=2,
        pool_size=2,
        dropout=0.1,
        context_length=100,
        pos_enc="relative",
        expansion_factor=3,
        separate_decoders=3,
        num_sequencing_platforms=2,
        num_runtypes=4
    )
    model = CANDI(
        signal_dim=25,
        metadata_embedding_dim=8,
        conv_kernel_size=3,
        n_cnn_layers=2,
        nhead=2,
        n_sab_layers=2,
        pool_size=2,
        dropout=0.1,
        context_length=100,
        pos_enc="relative",
        expansion_factor=3,
        separate_decoders=3,
        num_sequencing_platforms=2,
        num_runtypes=4
    )

    return model

def process_input_model(args):

    return make_inf_model(args.model_path)

def main():
    pass

if __name__=="__main__":
    main()
