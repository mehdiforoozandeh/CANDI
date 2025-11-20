"""
File to run model for CANDI inference

"""

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import json
import pysam
import torch
import pyBigWig
import numpy as np
import torch.nn.functional as F

from pred import CANDIPredictor # external import

# ------------------------------------------------------------------------
# Data loader
# ------------------------------------------------------------------------

class CANDI_INFERENCE:

    def __init__(self, temp_path, fasta_path, model : CANDIPredictor, debug):

        self.temp_path = temp_path
        self.load_fasta(fasta_path)

        self.model = model
        self.window = model.config.get('context-length', 1200)

        self.debug = debug

        self.chr_sizes_file = "./data/inf_debug_hg38.chrom.sizes"
        self.read_chr_sizes()

        # All possible assays
        self.ASSAYS=[
                'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac',
                'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3',
                'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac',
                'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac',
                'chipseq-control'
            ]

    def load_fasta(self, fasta_path):
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        if not os.path.exists(fasta_path + ".fai"):
            pysam.faidx(fasta_path)               # build index if needed
        self.fasta = pysam.FastaFile(fasta_path)  # pass path, not file handle

    def get_DNA_sequence(self, chrom, start, end):
        """
        Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

        :param fasta_file: Path to the fasta file.
        :param chrom: Chromosome name (e.g., 'chr1').
        :param start: Start position (0-based).
        :param end: End position (1-based, exclusive).
        :return: Sequence string.
        """
        # Ensure coordinates are within the valid range
        if start < 0 or end <= start:
            raise ValueError("Invalid start or end position")

        # Retrieve the sequence
        sequence = self.fasta.fetch(chrom, start, end)

        return sequence

    def dna_to_onehot(self, sequence):
        # Create a mapping from nucleotide to index
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}

        # Convert the sequence to indices
        indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)

        # Create one-hot encoding
        one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

        # Remove the fifth column which corresponds to 'N'
        one_hot = one_hot[:, :4]

        return one_hot

    def onehot_for_locus(self, locus):
        """
        Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
        Returns a tensor [context_length_bp, 4].
        """
        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        seq = self.get_DNA_sequence(chrom, start, end)
        if seq is None:
            return None
        return self.dna_to_onehot(seq)

    def load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as data:
            return data[data.files[0]]

    def load_json(self, file_name):
        # TODO: assert the order
        with open(file_name) as f:
            data = json.load(f)

        return data

    def read_chr_sizes(self):
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        self.chr_sizes_true = {}
        self.chr_sizes_candi = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes_true[chr_name] = int(chr_size)
                    self.chr_sizes_candi[chr_name] = int(chr_size) // 25 + 1

    # ------------------------------------------------------------------------
    # Data loading per chr
    # ------------------------------------------------------------------------

    def load_data_chr(self, chr_name):

        loaded_assays = {}

        for exp in os.listdir(self.temp_path):
            loaded_assays[exp] = {}

            exp_path = os.path.join(self.temp_path,exp)

            if not os.path.isdir(exp_path): continue

            chr_file = chr_name+".npz"
            npz_path = os.path.join(exp_path, chr_file)
            if not os.path.isfile(npz_path): continue
            assay_chr_data = self.load_npz(npz_path)

            loaded_assays[exp]["data"] = assay_chr_data

            input_metadata = self.load_json(os.path.join(exp_path, "input_metadata.json"))
            loaded_assays[exp]["input_metadata"] = list(input_metadata.values())

            output_metadata = self.load_json(os.path.join(exp_path, "output_metadata.json"))
            loaded_assays[exp]["output_metadata"] = list(output_metadata.values())

        return loaded_assays

    def make_full_tensor(self, chr, loaded_assays, missing_value=-1):

        dtensor = []
        input_mdtensor = []
        output_mdtensor = []
        availability = []

        missing_tensor = np.array([missing_value for _ in range(self.chr_sizes_candi[chr])])

        for exp in self.ASSAYS:

            if exp in loaded_assays.keys():
                # TODO if all missing consider it not available
                dtensor.append(loaded_assays[exp]["data"])
                availability.append(1)
                input_mdtensor.append(loaded_assays[exp]["input_metadata"])
                output_mdtensor.append(loaded_assays[exp]["output_metadata"])
            else:
                dtensor.append(missing_tensor)
                availability.append(0)
                # TODO: fix missing values
                input_mdtensor.append([missing_value, missing_value, missing_value, missing_value])
                output_mdtensor.append([missing_value, missing_value, missing_value, missing_value])

        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        input_mdtensor = torch.tensor(np.array(input_mdtensor)).permute(1, 0)
        output_mdtensor = torch.tensor(np.array(output_mdtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))

        return dtensor, input_mdtensor, output_mdtensor, availability

    def reshape_tensor(self, tensor, factor=1):

        n,a = tensor.shape

        window_new = self.window * factor
        # TODO ignore the last window
        pad_size = (window_new - n % window_new) % window_new
        padded_tensor = F.pad(tensor, (0, 0, 0, pad_size), mode='constant') # TODO: see if you can fix constant

        return padded_tensor.reshape(-1,window_new,a)

    def deshape_tensor(self, tensor, true_len):

        if self.debug: true_len = 2500
        b, l, f = tensor.shape
        tensor = tensor.reshape(b*l,f)
        return tensor[:true_len]

    def tensor_over_chr(self, chr):

        loaded_assays = self.load_data_chr(chr)
        L = self.chr_sizes_candi[chr]

        # TODO: overlapping
        dtensor, input_mdtensor, output_mdtensor, availability = self.make_full_tensor(chr, loaded_assays)
        dna_tensor = self.onehot_for_locus([chr, 0, L*25])

        if self.debug:
            dtensor = dtensor[:2500]
            dna_tensor = dna_tensor[:2500*25]

        dtensor = self.reshape_tensor(dtensor)
        dna_tensor = self.reshape_tensor(dna_tensor, 25)
        B,_,_ = dtensor.shape
        input_mdtensor = input_mdtensor.unsqueeze(0).float()
        input_mdtensor = input_mdtensor.repeat(B,1,1)
        output_mdtensor = output_mdtensor.unsqueeze(0).float()
        output_mdtensor = output_mdtensor.repeat(B,1,1)
        availability = availability.unsqueeze(0)
        availability = availability.repeat(B,1)

        return dtensor, input_mdtensor, output_mdtensor, availability, dna_tensor

    # ------------------------------------------------------------------------
    # Running through model
    # ------------------------------------------------------------------------

    def save_buffers(self, model_output):

        all_res = {
            "read_p" : model_output[0],
            "read_n" : model_output[1],
            "signal_mu" : model_output[2],
            "signal_var" : model_output[3],
            "peak_score" : model_output[4],
        }

        return all_res

    # def update_save_buffers(self, save_buffers, update):

    #     for i, val in enumerate(save_buffers.values()):
    #         val.append(update[i].numpy(force=True))

    # def stack_buffer(self, save_buffer):
    #     return np.hstack(save_buffer)[0]

    def save_chr_data(self, chr, save_buffers):

        for i, exp in enumerate(self.ASSAYS):
            if "control" in exp: continue

            exp_dir = os.path.join(self.temp_path, exp)
            os.makedirs(exp_dir, exist_ok=True)

            for key, val in save_buffers.items():
                values = val[:,i].numpy(force=True)
                save_dir = os.path.join(exp_dir, f"{chr}_{key}_predicted.npz")
                np.savez_compressed(save_dir, values)

    def actual_run(self):

        for chr in self.chr_sizes_true:

            dtensor, input_mdtensor, output_mdtensor, availability, dna_tensor = self.tensor_over_chr(chr)
            # TODO: batchsize.... cmd arg?
            # TODO: output mdtensor

            model_output = self.model.predict(
                    dtensor, input_mdtensor, output_mdtensor, availability, dna_tensor, []
                )

            model_output = [self.deshape_tensor(out_tensor, self.chr_sizes_candi[chr]) for out_tensor in model_output]
            all_res = self.save_buffers(model_output)
            self.save_chr_data(chr, all_res)

    # ------------------------------------------------------------------------
    # making bws
    # ------------------------------------------------------------------------

    def make_bws(self):

        bw_header = [(chr, chr_sizes) for chr, chr_sizes in self.chr_sizes_true.items()]

        for exp in self.ASSAYS:
            if "control" in exp: continue

            exp_dir = os.path.join(self.temp_path, exp)

            bw_read = pyBigWig.open(os.path.join(exp_dir, f"{exp}_read_mean.bw"), "w")
            bw_read.addHeader(bw_header)
            bw_signal = pyBigWig.open(os.path.join(exp_dir, f"{exp}_signal_mean.bw"), "w")
            bw_signal.addHeader(bw_header)
            bw_peak = pyBigWig.open(os.path.join(exp_dir, f"{exp}_peak_mean.bw"), "w")
            bw_peak.addHeader(bw_header)

            for chr in self.chr_sizes_true:

                data_read_p = self.load_npz(os.path.join(exp_dir, f"{chr}_read_p_predicted.npz"))
                data_read_n = self.load_npz(os.path.join(exp_dir, f"{chr}_read_n_predicted.npz"))
                data_signal_mu = self.load_npz(os.path.join(exp_dir, f"{chr}_signal_mu_predicted.npz"))
                data_signal_var = self.load_npz(os.path.join(exp_dir, f"{chr}_signal_var_predicted.npz"))
                data_peak_score = self.load_npz(os.path.join(exp_dir, f"{chr}_peak_score_predicted.npz"))

                data_read_mu =  (data_read_n * (1-data_read_p))/data_read_p

                bw_read.addEntries(chr, 0, values=data_read_mu, span=25, step=25)
                bw_signal.addEntries(chr, 0, values=data_signal_mu, span=25, step=25)
                bw_peak.addEntries(chr, 0, values=data_peak_score, span=25, step=25)

# ------------------------------------------------------------------------
# Main function call
# ------------------------------------------------------------------------

def run_through_model(args, model):


    fasta_path = os.fspath("/home/azr/lab/candix/EpiDenoise/data/hg38.fa")  # TODO: ask for this path
    inf_inst = CANDI_INFERENCE(args.temp_path, fasta_path, model, args.debug)

    inf_inst.actual_run()
    inf_inst.make_bws()

# ------------------------------------------------------------------------
# Extras
# ------------------------------------------------------------------------

def main():
    pass

if __name__=="__main__":
    main()
