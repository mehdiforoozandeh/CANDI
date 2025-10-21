"""
Main file or CANDI inference

"""

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import argparse
from inputs import process_input_data, process_input_model

def inf_arg_parser():
    """
    Argument parser of CANDI inference
    """

    parser = argparse.ArgumentParser(
        description="CANDI: Context-Aware Neural Data Imputation - Modern Inference Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                TODO: add examples here
                    """)

    # === DATA CONFIGURATION ===
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_path', type=str, default="./",
                            help='Path to the input folder.\n' \
                            'Folder should be set as follows:\n' \
                            './<BIO_SAMPLE>/<ASSAY>/<BAM>+<INPUT_MD>.TODO: make this better')
    data_group.add_argument('--model_path', type=str, default="./",
                            help='Path to the model')
    data_group.add_argument('--output_path', type=str, default="./",
                            help='Path to the outputs')
    data_group.add_argument('--tmp_path', type=str, default="./tmp",
                            help='Path to the temporary folder to save intermediate work.' \
                            'Deleted afterwards')

    # === SYSTEM CONFIGURATION ===
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified')
    system_group.add_argument('--check_gpus', action='store_true',
                             help='Check GPU availability and exit')

    # === CONFIGURATION FILE ===
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, default=None,
                             help='Path to YAML/JSON configuration file')
    config_group.add_argument('--save-config', type=str, default=None,
                             help='Save current configuration to file')

    # === ADVANCED OPTIONS ===
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--dsf-list', type=int, nargs='+', default=[1, 2],
                               help='Downsampling factors to use')
    advanced_group.add_argument('--debug', '-D', action='store_true',
                               help='Enable debug mode with extra logging')

    return parser.parse_args()

def main():

    args = inf_arg_parser()

    if not args.debug: process_input_data(args)

    model = process_input_model(args)

    return

if __name__=="__main__":
    main()

    # TODO: remove this
    # download_url = f"https://www.encodeproject.org/files/ENCFF282ZSZ/@@download/ENCFF282ZSZ.bam"
    # save_dir_name = "/home/azr/misc/test.bam"
    # import requests

    # with requests.get(download_url, stream=True) as response:
    #     response.raise_for_status()  # Check for request errors
    #     with open(save_dir_name, 'wb') as file:
    #         # Iterate over the response in chunks (e.g., 8KB each)
    #         for chunk in response.iter_content(chunk_size=int(1e3*1024)):
    #             # Write each chunk to the file immediately
    #             file.write(chunk)
