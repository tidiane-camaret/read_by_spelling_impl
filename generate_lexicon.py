import re
import unicodedata
import pickle
import random
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
         string_len=30,
         max_lexicon_len=300000,
         imgs_dataset_perc=0.3,
         stride = 10):

    PATH = path
    STRING_LEN = string_len


    lines = open(PATH + 'eng-fra.txt', encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs, normalize, get only the first one
    lexicon = [normalizeString(l.split('\t')[0]) for l in lines]

    # Delete multiple spaces
    lexicon = [" ".join(line.split()) for line in lexicon]

    lexicon = [x for x in lexicon if len(x) >= STRING_LEN]

    # fuse into string
    lexicon = ' '.join(lexicon)

    if max_lexicon_len > len(lexicon) // STRING_LEN:
        print("raw text can only produce non overlapping dataset of size", len(lexicon) // STRING_LEN)

    sentences = []
    nb_sentences = 0
    start = 0
    end = STRING_LEN
    while start < len(lexicon) and nb_sentences < max_lexicon_len:
        start = start + stride #lexicon[end:end+30].find(' ')
        end = start + STRING_LEN
        sentences.append(lexicon[start:end])
        nb_sentences += 1

    print("total number of strings : ", len(sentences))

    imgs_lexicon = sentences[:int(len(sentences) * imgs_dataset_perc)]
    exemples_lexicon = sentences[int(len(sentences)*imgs_dataset_perc):len(sentences)]

    print("number of strings for image generation: ", len(imgs_lexicon))
    print("number of strings for adversarial exemples: ", len(exemples_lexicon))


    with open(PATH + 'imgs_strings.pkl', 'wb') as f:
        pickle.dump(imgs_lexicon, f)

    with open(PATH + 'exemples_strings.pkl', 'wb') as f:
        pickle.dump(exemples_lexicon, f)

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-p', '--path',
                                default="data/lexicons/translation_dataset/",
                                help='Datasets path (write and read)',
                                type=str)
    cmdline_parser.add_argument('-sl', '--string_len',
                                default="30",
                                help='Length of generated strings',
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()

    generate_lexicon(args.path)
