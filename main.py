from generate_lexicon import generate_lexicon
from generate_dataset import generate_dataset
from train import train
import argparse

def run_training(lex_path,
                 imgs_path,
                 string_len,
                 dataset_size,
                 do_generate_lexicon,
                 do_generate_dataset,
                 verbose
                 ):
    # create lexicon
    if do_generate_lexicon:
        generate_lexicon(lex_path, string_len)

    #create image dataset
    if do_generate_dataset:
        generate_dataset(lex_path,
                 imgs_path,
                 string_len,
                 dataset_size)

    # train model on dataset
    train(lex_path,
          imgs_path,
          dataset_size,
          string_len,
          verbose=verbose
          )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-lp', '--lex_path',
                                default="data/lexicons/translation_dataset/",
                                help='lexicon path (write and read)',
                                type=str)
    cmdline_parser.add_argument('-ip', '--imgs_ds_path',
                                default="data/imgs/translation_dataset/",
                                help='images dataset path (write and read)',
                                type=str)
    cmdline_parser.add_argument('-sl', '--string_len',
                                default="30",
                                help='Length of generated strings',
                                type=int)
    cmdline_parser.add_argument('-il', '--imgs_ds_len',
                                default=50000,
                                help='images dataset length',
                                type=int)
    cmdline_parser.add_argument('-gl', '--gen_lex',
                                default=False,
                                help='generate lexicon from raw text ?',
                                type=bool)
    cmdline_parser.add_argument('-gi', '--gen_imgs_ds',
                                default=False,
                                help='generate image dataset from lexicon ?',
                                type=bool)
    cmdline_parser.add_argument('-v', '--verbose',
                                default=True,
                                help='print results at every batch ?',
                                type=bool)


    args, unknowns = cmdline_parser.parse_known_args()
    run_training(args.lex_path,
                 args.imgs_ds_path,
                 args.string_len,
                 args.imgs_ds_len,
                 args.gen_lex,
                 args.gen_imgs_ds,
                 args.verbose)

