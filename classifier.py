# EECS 486 - Final Project/classifer.py
# Pranav Ajith, Daniel Chandross, Tyler Eastman, Josie Elordi, Rana Makki

import os
import re
import sys
from preprocess import load_data, split_data, preprocess

def main(in_file):
    df = load_data(in_file)
    features, labels = split_data(df)
    preprocess(features)


if __name__ == '__main__':
	main(sys.argv[1])
