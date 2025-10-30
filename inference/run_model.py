"""
File to run model for CANDI inference

"""

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import pysam
import torch
import pyBigWig
import numpy as np

from pred import CANDIPredictor

# TODO: don't forget to remove test_exp and add a control
ASSAYS=[
        'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac', 
        'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3', 
        'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac', 
        'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac', 'control'
    ]

def load_fasta():
    fasta_path = os.fspath("/home/azr/lab/candix/EpiDenoise/data/hg38.fa")  # TODO: ask for this path
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if not os.path.exists(fasta_path + ".fai"):
        pysam.faidx(fasta_path)               # build index if needed
    return pysam.FastaFile(fasta_path)  # pass path, not file handle

def get_DNA_sequence(fasta, chrom, start, end):
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
    sequence = fasta.fetch(chrom, start, end)
    
    return sequence

def dna_to_onehot(sequence):
    # Create a mapping from nucleotide to index
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}
    
    # Convert the sequence to indices
    indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)
    
    # Create one-hot encoding
    one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

    # Remove the fifth column which corresponds to 'N'
    one_hot = one_hot[:, :4]
    
    return one_hot

def onehot_for_locus(fasta, locus):
    """
    Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
    Returns a tensor [context_length_bp, 4].
    """
    chrom, start, end = locus[0], int(locus[1]), int(locus[2])
    seq = get_DNA_sequence(fasta, chrom, start, end)
    if seq is None:
        return None
    return dna_to_onehot(seq)

def load_npz(file_name):
    with np.load(file_name, allow_pickle=True) as data:
    # with np.load(file_name, allow_pickle=True, mmap_mode='r') as data:
        return data[data.files[0]]
    
def load_data(bios_path):

    loaded_assays = {}
    chr_dict = {}

    for assay in os.listdir(bios_path):

        if not os.path.isdir(os.path.join(bios_path,assay)): continue

        if assay not in loaded_assays:
            loaded_assays[assay] = {}

        dsf_path = os.path.join(bios_path,assay,"signal_DSF1_res25")

        for chr_file in os.listdir(dsf_path):

            if "npz" in chr_file:
                
                chr_name = chr_file[:-4]

                npz_path = os.path.join(dsf_path, chr_file)
                chr_data = load_npz(npz_path)
                chr_len = chr_data.shape[0]

                if chr_name not in chr_dict:
                    chr_dict[chr_name] = chr_len
                else:
                    assert chr_dict[chr_name] == chr_len

                loaded_assays[assay][chr_name] = chr_data

    return loaded_assays, chr_dict
    
def make_full_tensor_bios(loaded_assays, chr, start=0, window=25000, missing_value=-1):
    dtensor = []
    mdtensor = []
    availability = []

    missing_tensor = np.array([missing_value for _ in range(window)])

    for assay in ASSAYS:
        
        if assay in loaded_assays.keys():
            dtensor.append(loaded_assays[assay][chr][start:start+window])
            availability.append(1)
            mdtensor.append([missing_value, missing_value, missing_value, missing_value])
        else:
            dtensor.append(missing_tensor)
            availability.append(0)
            mdtensor.append([missing_value, missing_value, missing_value, missing_value])
    
    dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
    mdtensor = torch.tensor(np.array(mdtensor)).permute(1, 0)
    availability = torch.tensor(np.array(availability))
    return dtensor, mdtensor, availability

def actual_run(args, data, chr_dict, model : CANDIPredictor, fasta):

    context_length = model.config.get('context-length', 1200)

    for chr in chr_dict.keys():

        L = chr_dict[chr]
        # TODO: padd data up to multiple of window

        output_p_complete = []
        output_n_complete = []
        output_mu_complete = []
        output_var_complete = []
        output_peak_complete = []

        for i in range(0,L,context_length):

            if args.debug and i > 10000: break

            dtensor, mdtensor, availability = make_full_tensor_bios(data, chr, i, context_length)
            dna_tensor = onehot_for_locus(fasta, [chr, i*25, (i+context_length)*25])

            dtensor = dtensor.unsqueeze(0).float()
            mdtensor = mdtensor.unsqueeze(0).float()
            availability = availability.unsqueeze(0)
            dna_tensor = dna_tensor.unsqueeze(0).float()

            output_p, output_n, output_mu, output_var, output_peak = model.predict(
                    dtensor, mdtensor, mdtensor, availability, dna_tensor , []
                )
            
            output_p_complete.append(output_p.numpy(force=True))
            output_n_complete.append(output_n.numpy(force=True))
            output_mu_complete.append(output_mu.numpy(force=True))
            output_var_complete.append(output_var.numpy(force=True))
            output_peak_complete.append(output_peak.numpy(force=True))
            
            pass

        output_p_complete = np.hstack(output_p_complete)[0]
        output_n_complete = np.hstack(output_n_complete)[0]
        output_mu_complete = np.hstack(output_mu_complete)[0]
        output_var_complete = np.hstack(output_var_complete)[0]
        output_peak_complete = np.hstack(output_peak_complete)[0]


        temp_path = args.temp_path
        bios_path = os.path.join(temp_path, os.listdir(temp_path)[0])

        for exp in os.listdir(bios_path):
            #TODO: make a dict of bw files
            bw = pyBigWig.open(os.path.join(bios_path, f"{exp}_signal_mean.bw"), "w")
            bw.addHeader([(chr, L*25)])
            exp_index = ASSAYS.index(exp)
            values = output_mu_complete[:,exp_index]
            bw.addEntries(chr, 0, values=values, span=25, step=25)
            bw.close()


def run_through_model(args, model):
    

    fasta = load_fasta()
    temp_path = args.temp_path
    bios_path = os.path.join(temp_path, os.listdir(temp_path)[0])

    avail_data, chr_dict = load_data(bios_path)

    actual_run(args, avail_data, chr_dict, model, fasta)

    pass

def main():
    pass

if __name__=="__main__":
    main()
