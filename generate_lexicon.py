import re
import unicodedata
import pickle
import random
import argparse
import collections

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
         max_lexicon_len=1000000,
         imgs_dataset_perc=0.2,
         stride = 10):

    PATH = path
    STRING_LEN = string_len


    lines = open(PATH + 'eng-fra.txt', encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs, normalize, get only the first one
    lexicon = [normalizeString(l.split('\t')[0]) for l in lines]

    # Delete multiple spaces
    lexicon = [" ".join(line.split()) for line in lexicon]

    lexicon_less = [x for x in lexicon if STRING_LEN - 5 <= len(x) <= STRING_LEN]
    lexicon_more = [x for x in lexicon if STRING_LEN < len(x)]

    for i in range(len(lexicon_less)):
        if len(lexicon_less[i]) < STRING_LEN:
            lexicon_less[i] = lexicon_less[i] + (STRING_LEN - len(lexicon_less[i])) * " "

     #fuse into string
    fused_lexicon = ' '.join(lexicon_more)

    lex = fused_lexicon.split(" ")
    print(max([len(l) for l in lex]))
    counter = collections.Counter(lex)

    print("vocabulary len : ", len(counter))

    lex = [l for l in lex if counter[l] > 10]

    print("new vocabulary len : ", len(collections.Counter(lex)))



    fused_lexicon = ' '.join(lex)

    print(len(lexicon_less))

    #if max_lexicon_len > len(fused_lexicon) // STRING_LEN:
    #    print("raw text can only produce non overlapping dataset of size", len(lexicon) // STRING_LEN)

    lexicon = []
    nb_sentences = 0
    start = 0
    end = STRING_LEN
    while end < len(fused_lexicon) - STRING_LEN and nb_sentences < max_lexicon_len:
        start = start + fused_lexicon[start:end].find(' ') + 1
        end = start + min(fused_lexicon[start:start + STRING_LEN].rfind(' '), STRING_LEN)
        lexicon.append(fused_lexicon[start:end])
        nb_sentences += 1


    for i in range(len(lexicon)):
        if len(lexicon[i]) < STRING_LEN:
            lexicon[i] = lexicon[i] + (STRING_LEN - len(lexicon[i])) * " "

    lexicon = lexicon + lexicon_less

    print("total number of strings : ", len(lexicon),
          "total number of unique strings", len(set(lexicon)))

    random.shuffle(lexicon)

    print(lexicon[:100])

    imgs_lexicon = lexicon[:int(len(lexicon) * imgs_dataset_perc)]
    exemples_lexicon = lexicon[int(len(lexicon)*imgs_dataset_perc):len(lexicon)]

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
