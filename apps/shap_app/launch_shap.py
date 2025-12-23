#!/usr/bin/env python3
"""
Launcher for SHAP analysis app. Downloads Zenodo results if needed and opens the Streamlit app.
"""
import argparse
import os
import sys
import subprocess

repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

ZENODO_URL = "https://zenodo.org/record/17935028/files/"

def download_example_files(target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    # This is a placeholder: user can download files manually from Zenodo
    print("Please download the Zenodo package from:")
    print("https://zenodo.org/records/17935028")
    print(f"and extract results into {target_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/", help="Directory for Zenodo results and CSVs")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        download_example_files(args.data_dir)

    print("To run SHAP UI, start Streamlit with the provided shap GUI script:")
    print("streamlit run shap_gui/shapey.py")

if __name__ == '__main__':
    main()
