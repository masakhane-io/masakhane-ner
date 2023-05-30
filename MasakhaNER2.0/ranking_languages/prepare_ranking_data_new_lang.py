import os
from transformers import AutoTokenizer
import pandas as pd

model_name_or_path = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def extract_sentences(input_file):
    with open(input_file) as f:
        sentences = f.readlines()

    return sentences

def export_sentences(exp_dir, sentences, file):
    with open(exp_dir+file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent+'\n')


def export_spm_sentences(exp_dir, sentences, file):
    with open(exp_dir+file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            tok_sents = ' '.join(tokenizer.tokenize(sent))
            f.write(tok_sents+'\n')


if __name__ == "__main__":

    output_dir = 'ranking_data/datasets/'
    output_spm_dir = 'ranking_data/datasets_spm/'
    create_dir(output_dir)
    create_dir(output_spm_dir)
    for lang in ['ful', 'lin', 'orm', 'sot', 'ven', 'run', 'tir']:
        print(lang)
        sentences = extract_sentences("ranking_data/datasets/ner-train.orig." + lang)
        export_spm_sentences(output_spm_dir, sentences, 'ner-train.orig.spm.' + lang)


