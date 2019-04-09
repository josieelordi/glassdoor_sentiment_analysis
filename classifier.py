# Josie Elori, Tyler Eastman, Daniel Chandross, Pranav Ajith, Rana Makki

import os
import re
import sys
import preprocess.py
import spacy
import string
import pandas as pd

if __name__ == '__main__':
	print("BIOOOOOOTCH")

	in_file = sys.argv[1]
	print(in_file)

	df = load_data(in_file)

	print(df)