## Imports

# System
import os
import sys

# Set current and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the current path
sys.path.append(parent_dir)

# Data Analysis
import pandas as pd
from data_preprocessing_packages.feature_extraction import extract_feature

# Importing Data

# Add the parent directory to the current path
sys.path.append(parent_dir)

## Script for running the experiment.

# Import Dataframes
positive = pd.read_csv('data/Positive data.csv')
negative = pd.read_csv('data/Negative data.csv')

# Extract Features
positive_data = extract_feature(positive['Seq'])
negative_data = extract_feature(negative['Seq'])

# Add targets
positive_data['Class'] = 1
negative_data['Class'] = 0

positive_data['SeqLength_bins'] = pd.qcut(positive_data['SeqLength'], q=5)
negative_data['SeqLength_bins'] = pd.qcut(positive_data['SeqLength'], q=5)

full_data = pd.concat([positive_data, negative_data])

# We only care about the class
peptide_classes = full_data['Class']

