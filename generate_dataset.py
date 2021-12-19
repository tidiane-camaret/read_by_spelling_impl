import os
import argparse

import pickle
import string
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter


from skimage.util import random_noise



class create_string_to_image_dataset:
    def __init__(self, lexicon: list,
                 nb_ex: int,
                 img_size: tuple = (250, 25),
                 font_names: list = ["FreeSerifItalic.ttf"],
                 blur: float = None,
                 noise: float = None,
                 write: string = None,
                 draw_rectangles: bool = False):

        self.labels = []
        self.imgs = []
        word_count = 0

        for string in lexicon:
            #string = re.sub(r'\W+ ', '', string)
            for font_name in font_names:
                img, label = string_to_image(string,
                                word_count,
                                font_name,
                                blur,
                                noise,
                                draw_rectangles)

                if write:
                    img.save(write +"img"+ str(word_count) + ".png")

                word_count += 1
                self.labels.append(label)
                self.imgs.append(img)

                if word_count > nb_ex:
                    break
            if word_count > nb_ex:
                break

        if write:
            with open(write+'img_labels.pkl', 'wb') as f:
                pickle.dump(self.labels, f)


def string_to_image(string,
                    word_count=0,
                    font_name="FreeSerifItalic.ttf",
                    blur: float = None,
                    noise: float = None,
                    draw_rectangles: bool = False,
                    fixed_lenght=None,
                    draw_char_by_char=False):

    font = ImageFont.truetype("data/fonts/" + font_name, 25)

    img = Image.new("RGBA", (1, 1))
    text_size = ImageDraw.Draw(img).textsize(string,
                                             font)  # we first write on another image to preview total text size
    lenght = (text_size[0] + 5, text_size[1] + 5)  # we add a margin at the end of the text
    if fixed_lenght: lenght = fixed_lenght

    img = Image.new("RGBA", lenght, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    label = []

    if draw_char_by_char:

        w, h = 0, 0

        for i in range(len(string)):
            draw.text((w, 0), string[i], (0, 0, 0), font=font)
            w_l, h_l = draw.textsize(string[i], font)
            label.append([string[i], word_count, w, 0, w + w_l, h_l])

            if draw_rectangles:
                draw.rectangle([(w, 0), (w + w_l, h_l)], outline="red")
            w += w_l
    else:
        draw.text((0, 0), string, (0, 0, 0), font=font)
        for i in range(len(string)):
            label.append([string[i], word_count, None, None, None, None])

    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    if noise:
        img_arr = np.asarray(img)
        noise_img = random_noise(img_arr, mode='gaussian', var=noise ** 2)
        noise_img = (255 * noise_img).astype(np.uint8)
        img = Image.fromarray(noise_img)

    return img, label

def img_to_croppings(img,
                     locations):

    img_list = []
    for loc in locations:
        img_crop = img.crop(tuple(loc))
        img_list.append(img_crop)

    return img_list

def generate_dataset(lex_path,
         imgs_path,
         string_len=30,
         dataset_size=100000,
         ):

    STRING_LEN = string_len
    DATASET_SIZE = dataset_size

    DATASET_DIR = imgs_path
    os.makedirs(DATASET_DIR, exist_ok=True)
    lexfilename = lex_path + "imgs_strings.pkl"#'/media/tidiane/D:/Dev/CV/unsupervised_ocr/data/translation_dataset/translation_ds_strings.pkl'


    with open(lexfilename, 'rb') as f:
        lexicon = pickle.load(f)

    print(len(lexicon))
    lexicon = [x[0:STRING_LEN] for x in lexicon if len(x) >= STRING_LEN]
    print(lexicon[0:10])
    print(len(lexicon))


    font_names = ["FreeSerifItalic.ttf", "FreeMonoBoldOblique.ttf", "FreeSansBoldOblique.ttf"]# "FreeSerifBoldItalic.ttf"]#,
                  #"FreeMonoBold.ttf", "FreeSansBold.ttf", "FreeSerifBold.ttf", "FreeMonoOblique.ttf", "FreeSansOblique.ttf",
                  #"FreeSerifItalic.ttf", "FreeMono.ttf", "FreeSans.ttf", "FreeSerif.ttf"]


    lexicon = lexicon[0:DATASET_SIZE*len(font_names)]

    #generate dataset
    string_to_image_dataset = create_string_to_image_dataset(lexicon=lexicon,
                                                             nb_ex=DATASET_SIZE,
                                                             font_names=font_names,
                                                             blur=0.1,
                                                             noise=0.25,
                                                             write=DATASET_DIR)

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
    cmdline_parser.add_argument('-dl', '--dataset_len',
                                default="5000",
                                help='Length of generated dataset',
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()

    generate_dataset(args.path,
         args.string_len,
         args.dataset_len)
