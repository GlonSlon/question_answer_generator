# Automatic Question-Answer Generator from Text Files

A Python tool that automatically generates question-answer pairs from text documents using NLP techniques and machine learning. The system extracts key information from `.txt` files and creates QA pairs suitable for training datasets or knowledge bases.

## Features

- **Multi-file processing**: Reads all `.txt` files in the current directory (except `dataset_gpt.txt`)
- **Intelligent sentence analysis**: Identifies definition sentences and key concepts
- **Question generation**: Creates relevant questions based on sentence content
- **ML-enhanced ranking**: Uses TF-IDF to prioritize important sentences (when scikit-learn is available)
- **Deduplication**: Removes duplicate questions and filters low-quality pairs
- **Output formats**: Generates both a simple text dataset and a JSON metadata file

## Requirements

- Python 3.7+
- Dependencies (optional but recommended):
  - `nltk`: Improved sentence tokenization
  - `spacy`: Better key term extraction (with en_core_web_sm model)
  - `scikit-learn`: TF-IDF based sentence importance ranking
  - `tqdm`: Progress bars

## Install the required dependencies:
```bash
  pip install nltk spacy scikit-learn tqdm
```
## Download the spaCy English model (if using spaCy):
```bash
  python -m spacy download en_core_web_sm
```
## Usage:
Place your .txt files in the same directory as the script.
```bash
  python qa_generator.py
```
## The script will:
- Process all .txt files (excluding dataset_gpt.txt)
- Generate question-answer pairs
- Save results to dataset_gpt.txt and dataset_gpt_meta.json
    
