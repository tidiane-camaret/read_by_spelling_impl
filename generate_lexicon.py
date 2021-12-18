import re
import unicodedata
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z ]+", r" ", s)
    return s

# Lowercase, trim, and remove non-letter characters

def generate_lexicon(path,
         string_len=30):
    PATH = path
    STRING_LEN = string_len



    lines = open(PATH + 'eng-fra.txt', encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    lexicon = [normalizeString(l.split('\t')[0]) for l in lines]

    lexicon = [" ".join(line.split()) for line in lexicon]
    lexicon = [x[0:STRING_LEN].lower() for x in lexicon if len(x) >= STRING_LEN]

    print(len(lexicon))
    with open(PATH + 'strings.pkl', 'wb') as f:
        pickle.dump(lexicon[:len(lexicon)//2], f)

    with open(PATH + 'strings2.pkl', 'wb') as f:
        pickle.dump(lexicon[len(lexicon)//2:], f)

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-p', '--path',
                                default="data/translation_dataset/",
                                help='Datasets path (write and read)',
                                type=str)
    cmdline_parser.add_argument('-sl', '--string_len',
                                default="30",
                                help='Length of generated strings',
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()

    generate_lexicon(args.path)
