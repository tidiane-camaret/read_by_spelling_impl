from generate_lexicon import generate_lexicon
from generate_dataset import generate_dataset
from train import train
import argparse

def run_training(path,
                 string_len,
                 dataset_size,
                 do_generate_lexicon,
                 do_generate_dataset,
                 ):
    # create lexicon
    if do_generate_lexicon:
        generate_lexicon(path, string_len)

    #create image dataset
    if do_generate_dataset:
        generate_dataset(path, string_len, dataset_size)

    # train model on dataset
    train(path,
          dataset_size,
          string_len,
          )

# Press the green button in the gutter to run the script.
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
    cmdline_parser.add_argument('-d', '--dataset_len',
                                default=500000,
                                help='dataset length',
                                type=int)
    cmdline_parser.add_argument('-gl', '--gen_lex',
                                default=False,
                                help='generate lexicon at path ?',
                                type=bool)
    cmdline_parser.add_argument('-gd', '--gen_ds',
                                default=False,
                                help='generate dataset at path ?',
                                type=bool)

    args, unknowns = cmdline_parser.parse_known_args()
    run_training(args.path,
                 args.string_len,
                 args.dataset_len,
                 args.gen_lex,
                 args.gen_ds)

