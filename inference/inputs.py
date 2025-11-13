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
from data_utils import BAM_TO_SIGNAL # external import
from pred import CANDIPredictor # external import

# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
def process_bam(bam_file):
    # Process BAM to signals using existing BAM_TO_SIGNAL

    bam_processor = BAM_TO_SIGNAL(
        bam_file=bam_file,
        chr_sizes_file="./data/inf_debug_hg38.chrom.sizes" #TODO: know where to get these sizes from
    )
    bam_processor.full_preprocess(dsf_list=[1])

    os.remove(bam_file)
    if os.path.exists(f"{bam_file}.bai"):
        os.remove(f"{bam_file}.bai")

def process_input_data(base_path, temp_path):

    bio_samples = os.listdir(base_path)

    for bio_sample in bio_samples:

        bio_sample_path = os.path.join(base_path, bio_sample)
        temp_bio_sample_path = os.path.join(temp_path, bio_sample)
        os.makedirs(temp_bio_sample_path, exist_ok=True)

        exps = os.listdir(bio_sample_path)

        for exp in exps:

            exp_path = os.path.join(bio_sample_path, exp)
            temp_exp_path = os.path.join(temp_bio_sample_path, exp)
            os.makedirs(temp_exp_path, exist_ok=True)

            for file in os.listdir(exp_path):

                if file.endswith(".bam"):

                    file_path = os.path.join(exp_path,file)
                    temp_file_path = os.path.join(temp_exp_path,file)
                    shutil.copy2(file_path, temp_file_path)
                    pysam.index(temp_file_path)

                    with contextlib.redirect_stdout(None):
                        process_bam(temp_file_path)

# ------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------

def load_candi_predictor(model_path):

    return CANDIPredictor(model_path)

# ------------------------------------------------------------------------
# Extra stuff
# ------------------------------------------------------------------------

def main():
    pass

if __name__=="__main__":
    main()
