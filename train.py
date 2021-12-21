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

def train(lex_path,
          imgs_path,
         dataset_max_len=500000,
         string_len=30,
         batch_size=64,
         n_epochs = 100,
         embed_size=256,
         nb_filters=512,
         save_model=True,
         verbose=True,
         ):


    LEXICON_FILE_PATH = lex_path + "exemples_strings.pkl"
    DATASET_PATH = imgs_path
    DATASET_MAX_LEN = dataset_max_len
    STRING_LEN = string_len
    VOC_LIST = list(ascii_lowercase + ' ')
    VOC_LEN = len(VOC_LIST)
    NB_SPE_CHAR = 3  # sos, pad, unk

    EMBED_SIZE = embed_size
    NB_FILTERS = nb_filters

    print_every = 20

    cuda = True if torch.cuda.is_available() else False

    print("cuda : ", cuda)

    # !!! Minimizes MSE instead of BCE
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(STRING_LEN, voc_len=VOC_LEN+NB_SPE_CHAR)#.to(device)
    discriminator = DiscriminatorMSE(STRING_LEN, VOC_LEN+NB_SPE_CHAR, EMBED_SIZE, NB_FILTERS)#.to(device)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    #optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.001)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    dataset = string_img_Dataset(img_size=(32, STRING_LEN*2**4),
                                 max_len=DATASET_MAX_LEN,
                                 string_tensor_length=STRING_LEN,
                                 voc_list=ascii_lowercase + ' ',
                                 dataset_dir=DATASET_PATH)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    # ----------
    #  Training
    # ----------

    results = []

    with open(LEXICON_FILE_PATH, 'rb') as f:
        lexicon = pickle.load(f)

    for epoch in range(n_epochs):

        for i, (imgs, targets) in enumerate(train_loader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            if cuda:
                valid = valid.cuda()
                fake = fake.cuda()
                imgs = imgs.cuda()
                targets = targets.cuda()


            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            sampled_indexes = np.random.randint(0, len(lexicon), batch_size)
            real_imgs = [string_to_tensor(lexicon[x], STRING_LEN, voc_list=VOC_LIST) for x in sampled_indexes]
            real_imgs = torch.stack(real_imgs)

            if cuda:
                real_imgs = real_imgs.cuda()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()


            if verbose :


                output = tensor_to_string(gen_imgs[0].detach().cpu().numpy(), voc_list=VOC_LIST)
                target = tensor_to_string(targets[0].detach().cpu().numpy(), voc_list=VOC_LIST)
                exemple = tensor_to_string(real_imgs[0].detach().cpu().numpy(), voc_list=VOC_LIST)
                score = 0
                for l in range(len(target.rstrip())):
                    if output[l] == target[l]:
                        score += 1
                score = score/len(target.rstrip())

                results.append([epoch,
                                i,
                                g_loss.item(),
                                d_loss.item(),
                                target,
                                output,
                                score])

                if i % print_every == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]"
                        % (epoch, n_epochs, i, len(train_loader), g_loss.item(), d_loss.item()))

                    print("exemple : ", exemple)
                    print("target  : ", target)
                    print("output  : ", output, ", char acc = ", score)

        if save_model:
            torch.save(generator.state_dict(), "models_data/"+str(epoch)+"_gen.pt")
            torch.save(generator.state_dict(), "models_data/"+str(epoch)+"_disc.pt")
            with open("models_data/run_results.pkl", 'wb') as f:
                pickle.dump(results, f)

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
