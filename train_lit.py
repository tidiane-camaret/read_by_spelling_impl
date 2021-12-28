"""
Inspired by LSGan implementation :
https://github.com/meliketoy/LSGAN.pytorch/blob/master/main.py
"""

import argparse
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable  # deprecated

from train_utils import string_img_Dataset
from models import *
from string import ascii_lowercase
import pytorch_lightning as pl

VOC_LIST = list(ascii_lowercase + ' ')
NB_SPE_CHAR = 3


class LitTransformerGan(pl.LightningModule):
    def __init__(self, string_len, voc_len, embed_size, nb_filters, lexicon):
        super().__init__()

        self.generator = Generator(string_len, voc_len=voc_len + NB_SPE_CHAR)
        self.discriminator = DiscriminatorMSE(string_len, voc_len + NB_SPE_CHAR, embed_size, nb_filters)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.criterion = torch.nn.MSELoss()
        self.lexicon = lexicon
        self.string_len = string_len

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        g_opt, d_opt = self.optimizers()

        images, _ = batch

        generated_labels = self.generator(images)

        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        batch_size = images.shape[0]

        real_label = torch.ones((batch_size), device=self.device)
        fake_label = torch.zeros((batch_size), device=self.device)

        sampled_indexes = np.random.randint(0, len(self.lexicon), batch_size)
        real_imgs = [string_to_tensor(self.lexicon[x], self.string_len, voc_list=VOC_LIST) for x in sampled_indexes]
        example_labels = torch.stack(real_imgs).type_as(images)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.discriminator(example_labels)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.discriminator(generated_labels.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = (errD_real + errD_fake)/2

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.discriminator(generated_labels)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        output = self.generator(images)
        loss = self.criterion(output, targets)
        # Logging to TensorBoard by default
        # self.log("val_loss", loss)

        output = tensor_to_string(output[0].detach().cpu().numpy(), voc_list=VOC_LIST)
        target = tensor_to_string(targets[0].detach().cpu().numpy(), voc_list=VOC_LIST)
        score = 0
        for l in range(min(len(target), len(output))):
            if output[l] == target[l]:
                score += 1
        acc = score / len(target.rstrip())

        self.log(f"val_loss", loss, prog_bar=False)
        self.log(f"acc", acc, prog_bar=False)

        if batch_idx % 10 == 0:
            print("output : ", output, "\n")
            print("target : ", target, "\n")

    def configure_optimizers(self):
        g_opt = torch.optim.RMSprop(self.generator.parameters(), lr=0.001)
        d_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.001)
        return g_opt, d_opt

def train(lex_path,
         images_path,
         dataset_max_len=500000,
         string_len=30,
         batch_size=8,
         n_epochs=100,
         embed_size=256,
         nb_filters=512,
         ):


    num_workers, num_gpus = (4, 1) if torch.cuda.is_available() else (0, 0)

    lex_path = lex_path + "exemples_strings.pkl"

    with open(lex_path, 'rb') as f:
        lexicon = pickle.load(f)

    transformer = LitTransformerGan(string_len=string_len,
                                    voc_len=len(VOC_LIST),
                                    embed_size=embed_size,
                                    nb_filters=nb_filters,
                                    lexicon=lexicon)

    trainer = pl.Trainer(max_epochs=n_epochs,
                         gpus=num_gpus
                         )
    dataset = string_img_Dataset(img_size=(32, string_len*2**4),
                                             batch_size=batch_size,
                                             max_len=dataset_max_len,
                                             string_tensor_length=string_len,
                                             voc_list=ascii_lowercase + ' ',
                                             dataset_dir=images_path,
                                             )

    ds_len = int(len(dataset) * 0.95)

    train_set, val_set = torch.utils.data.random_split(dataset, [ds_len, len(dataset) - ds_len])

    train_set = torch.utils.data.DataLoader(train_set,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        drop_last=True)
    val_set = torch.utils.data.DataLoader(val_set,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        drop_last=True)
    trainer.fit(transformer, train_set, val_set)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-p', '--path',
                                default="data/translation_dataset/",
                                help='Datasets path (write and read)',
                                type=str)
    cmdline_parser.add_argument('-d', '--dataset_len',
                                default=500000,
                                help='dataset length',
                                type=int)
    cmdline_parser.add_argument('-l', '--str_len',
                                default=30,
                                help='string_length',
                                type=int)

    args, unknowns = cmdline_parser.parse_known_args()

    train(args.path,
         dataset_max_len=args.dataset_len,
         string_len=args.str_len)
