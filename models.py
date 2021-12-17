import nltk

import torch
import torch.nn as nn

from train_utils import tensor_to_string, string_to_tensor, get_rand_strings_from_lexicon, get_rand_strings


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, string_len, voc_len):
        super(Generator, self).__init__()
        self.generator_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2)
        )
        self.generator_block_234 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2)
        )

        self.AvgPool = nn.AvgPool1d(kernel_size=2)
        self.Unflatten = nn.Unflatten(dim=1, unflattened_size=(string_len, 32))
        self.Linear = nn.Linear(in_features=32, out_features=voc_len)


    def forward(self, x):
        # initial shape : [B, D, H = 32, W = n * 16]
        x = self.generator_block_1(x)
        #print(x.shape) # shape : [B, D, 16, n * 8]

        for i in range(3):
            x = self.generator_block_234(x)
            #print(x.shape)

        # shape : [B, D, 2, n]
        x = torch.permute(x, (0, 3, 1, 2))
        #print(x.shape) # shape : [B, n, D, 2]
        x = torch.flatten(x,start_dim=1,end_dim=2)
        #print(x.shape) #shape : [B, n * D, 2]
        x = self.AvgPool(x)
        #print(x.shape)  # shape : [B, n * D, 1]
        x = torch.squeeze(x)
        #print(x.shape) # shape : [B, n * D]
        x = self.Unflatten(x)
        #print(x.shape)# shape : [B, n, D]
        x = self.Linear(x)
        #print(x.shape)  # shape : [B, n, K]

        return x

class DiscriminatorMSE(nn.Module):
    def __init__(self, string_len, voc_len, embed_size, nb_filters):
        super(DiscriminatorMSE, self).__init__()

        self.Softmax = nn.Softmax(dim=2)
        self.Embedding = nn.Linear(in_features=voc_len, out_features=embed_size)
        # nn.Embedding(embedding_dim=1, num_embeddings=embed_size)
        self.Conv_1 = nn.Conv1d(in_channels=embed_size, out_channels=nb_filters, kernel_size=5, padding=2)
        self.LayerNorm = nn.LayerNorm(normalized_shape=[nb_filters, string_len])
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        self.discriminator_block = nn.Sequential(
            nn.Conv1d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=5, padding=2),
            nn.LayerNorm(normalized_shape=[nb_filters, string_len]),
            nn.LeakyReLU(negative_slope=0.2))

        self.Conv_2 = nn.Conv1d(in_channels=nb_filters, out_channels=embed_size, kernel_size=5, padding=2)
        self.LayerNorm_2 = nn.LayerNorm(normalized_shape=[embed_size, string_len])
        self.LeakyReLU_2 = nn.LeakyReLU(negative_slope=0.2)

        self.LinearFlattened = nn.Linear(string_len*embed_size, 100)
        self.LinearFlattened2 = nn.Linear(100, 2)


        self.Linear = nn.Linear(in_features=embed_size, out_features=1)
        self.LeakyReLU_3 = nn.LeakyReLU(negative_slope=0.2)
        self.LinearFinal = nn.Linear(in_features=string_len, out_features=1)
        self.AvgPool = nn.AvgPool1d(kernel_size=string_len)
        self.Activ = nn.Tanh()

    def forward(self, x):

        x = self.Softmax(x)
        #print("softmax : ", x.shape)
        x = self.Embedding(x)
        #print("embedding : ", x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print("permute : ", x.shape)
        x = self.Conv_1(x)
        #print("conv1 : ", x.shape)
        x = self.LayerNorm(x)
        #print("layernorm : ", x.shape)
        x = self.LeakyReLU(x)
        #print("ReLU : ", x.shape)

        for i in range(3):
            x = self.discriminator_block(x)
            #print(x.shape)

        x = self.Conv_2(x)
        #print("conv : ", x.shape)
        x = self.LayerNorm_2(x)
        #print(x.shape)
        x = self.LeakyReLU_2(x)

        #print("relu : ",x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print("permute : ",x.shape)
        x = self.Linear(x)
        #print("linear : ", x.shape)
        x = torch.squeeze(x)
        #print("squeeze : ",x.shape)

        x = self.LeakyReLU_3(x)

        x = self.AvgPool(x)
        #print("avgpool : ",x.shape)
        x = torch.squeeze(x)
        #print("squeeze : ", x.shape)
        #x = self.Activ(x)


        return x

class Discriminator(nn.Module):
    def __init__(self, string_len, voc_len, embed_size, nb_filters):
        super(Discriminator, self).__init__()

        self.Softmax = nn.Softmax(dim=2)
        self.Embedding = nn.Linear(in_features=voc_len, out_features=embed_size)
        # nn.Embedding(embedding_dim=1, num_embeddings=embed_size)
        self.Conv_1 = nn.Conv1d(in_channels=embed_size, out_channels=nb_filters, kernel_size=5, padding=2)
        self.LayerNorm = nn.LayerNorm(normalized_shape=[nb_filters, string_len])
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        self.discriminator_block = nn.Sequential(
            nn.Conv1d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=5, padding=2),
            nn.LayerNorm(normalized_shape=[nb_filters, string_len]),
            nn.LeakyReLU(negative_slope=0.2))

        self.Conv_2 = nn.Conv1d(in_channels=nb_filters, out_channels=embed_size, kernel_size=5, padding=2)
        self.LayerNorm_2 = nn.LayerNorm(normalized_shape=[embed_size, string_len])
        self.LeakyReLU_2 = nn.LeakyReLU(negative_slope=0.2)

        self.LinearFlattened = nn.Linear(string_len*embed_size, 100)
        self.LinearFlattened2 = nn.Linear(100, 2)


        self.Linear = nn.Linear(in_features=embed_size, out_features=1)
        self.LeakyReLU_3 = nn.LeakyReLU(negative_slope=0.2)
        self.LinearFinal = nn.Linear(in_features=string_len, out_features=2)
        self.AvgPool = nn.AvgPool1d(kernel_size=string_len)
        self.Activ = nn.Tanh()

    def forward(self, x):

        x = self.Softmax(x)
        #print("softmax : ", x.shape)
        x = self.Embedding(x)
        #print("embedding : ", x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print("permute : ", x.shape)
        x = self.Conv_1(x)
        #print("conv1 : ", x.shape)
        x = self.LayerNorm(x)
        #print("layernorm : ", x.shape)
        x = self.LeakyReLU(x)
        #print("ReLU : ", x.shape)

        for i in range(3):
            x = self.discriminator_block(x)
            #print(x.shape)

        x = self.Conv_2(x)
        #print("conv : ", x.shape)
        x = self.LayerNorm_2(x)
        #print(x.shape)
        x = self.LeakyReLU_2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.LinearFlattened(x)
        x = self.LinearFlattened2(x)
        """
        #print("relu : ",x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print("permute : ",x.shape)
        x = self.Linear(x)
        #print("linear : ", x.shape)
        x = torch.squeeze(x)
        #print("squeeze : ",x.shape)

        x = self.LeakyReLU_3(x)

        #x = self.AvgPool(x)
        #print("avgpool : ",x.shape)
        x = torch.squeeze(x)
        #print("squeeze : ", x.shape)
        #x = self.Activ(x)
        x = self.LinearFinal(x)
"""
        return x

class CharacterLevelCNN(nn.Module):
    def __init__(self, n_classes=14, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CharacterLevelCNN, self).__init__()

        embed_size = 128

        self.Softmax = nn.Softmax(dim=2)
        self.Embedding = nn.Linear(in_features=input_dim, out_features=embed_size)

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=5, padding=2), nn.LeakyReLU(0.2),)
                                   #nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=5, padding=2), nn.LeakyReLU(0.2),)
                                   #nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=5, padding=2), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=5, padding=2), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=5, padding=2), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=5, padding=2), nn.LeakyReLU(0.2),)
                                   #nn.MaxPool1d(3))

        dimension = 7680#int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons))#, nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons))#, nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        # other end options
        self.fc_bis = nn.Linear(n_conv_filters, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.fc2_bis = nn.Linear(input_length, 2)




        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        #print("input : ", input.shape)

        #input = self.Softmax(input)
        #input = self.Embedding(input)

        #print("embedd : ", input.shape)

        input = input.transpose(1, 2)


        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        #print("convs : ",output.shape)

        # original
        #output = output.view(output.size(0), -1)
        #output = self.fc1(output)
        #output = self.fc2(output)
        #output = self.fc3(output)

        #other option
        output = output.transpose(1, 2)
        output = self.fc_bis(output)
        output = output.view(output.size(0), -1)
        output = self.leakyrelu(output)
        output = self.fc2_bis(output)

        return output


def run_test_batch(Gen,Disc, test_loader):
    for batch_idx, (img, target) in enumerate(test_loader):

        gen_output = Gen(img)
        #print(gen_output.size())

        gen_output = nn.Softmax(dim=2)(gen_output)
        lev_dist = 0

        for i in range(len(target)):
            out_string = tensor_to_string(gen_output.detach().numpy()[i])
            target_string = tensor_to_string(target[i])
            lev_dist += nltk.edit_distance(out_string, target_string)

            if i == 0:
                print(target_string, " : ", out_string)
        print("average edit dist : ", lev_dist/len(target))
        #disc_output = Disc(gen_output)
        # print(gen_output.shape)
        #print(disc_output.shape)



def train(Gen, Disc, device, train_loader, optimizerG, optimizerD,
          epoch, log_interval, dry_run, string_len, batch_size, voc_list):

    loss = nn.MSELoss()

    Gen.train
    Disc.train

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(target).to(device)
        #print(data.size())

        # zero the gradients on each iteration
        optimizerG.zero_grad()

        #Generated output
        gen_output = Gen(data)
        #print("gen output shape : ",gen_output.size())

        #get real data
        _, true_data = get_rand_strings_from_lexicon(string_len=string_len, batch_size=batch_size, )
        true_data = [string_to_tensor(x, string_len, voc_list=voc_list)
                     for x in true_data]
        true_labels = torch.ones((batch_size)).float()
        true_data = torch.stack(true_data)

        #print(true_data.shape)

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        gen_disc_output = Disc(gen_output)[:, 0]
        if gen_disc_output.size() != true_labels.size():
            print(gen_disc_output.size(), true_labels.size())
        generator_loss = loss(gen_disc_output, true_labels)
        generator_loss.backward()
        optimizerG.step()

        # Train the discriminator on the true data
        optimizerD.zero_grad()
        true_discriminator_out = Disc(true_data)[:, 0]
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # Train the discriminator on the generated data
        # add .detach() here think about this
        generator_discriminator_out = Disc(gen_output.detach())[:, 0]
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 4
        discriminator_loss.backward()
        optimizerD.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGen. Loss: {:.6f}, Discr. Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), generator_loss.item(), discriminator_loss.item()))
            #print("discriminator output on fake data : ", gen_disc_output)
            #print("discriminator output on real data : ", true_discriminator_out)
            print(tensor_to_string(gen_output[0].detach().numpy(), voc_list=voc_list))
            print(tensor_to_string(target[0].detach().numpy(), voc_list=voc_list))
            if dry_run:
                break


def train_CELoss(Gen, Disc, device, train_loader, optimizerG, optimizerD,
          epoch, log_interval, dry_run, string_len, batch_size, voc_list):

    loss = nn.CrossEntropyLoss()

    Gen.train
    Disc.train

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(target).to(device)

        # zero the gradients on each iteration
        optimizerG.zero_grad()

        #Generated output
        gen_output = Gen(data)
        #print("gen outpur shape : ",gen_output.size())

        #get real data
        true_labels, true_data = get_rand_strings_from_lexicon(string_len=string_len, batch_size=batch_size)
        true_data = [string_to_tensor(x, string_len, voc_list=voc_list)
                     for x in true_data]

        true_data = torch.stack(true_data)

        #true_labels = torch.tensor(true_labels).float()
        true_labels = torch.empty(size=(batch_size, 2), device=device).float()
        for i in range(batch_size):
            true_labels[i] = torch.tensor([1, 0])

        false_labels = torch.empty(size=(batch_size, 2),device=device).float()
        for i in range(batch_size):
            false_labels[i] = torch.tensor([0, 1])

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        gen_disc_output = Disc(gen_output)
        #print(gen_output.size())
        if gen_disc_output.size() != true_labels.size():
            print(gen_disc_output.size(), true_labels.size())
        generator_loss = loss(gen_disc_output, true_labels)
        generator_loss.backward()
        optimizerG.step()

        # Train the discriminator on the true data
        optimizerD.zero_grad()
        true_discriminator_out = Disc(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # Train the discriminator on the generated data
        # add .detach() here think about this
        generator_discriminator_out = Disc(gen_output.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, false_labels)
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 8
        #print(batch_idx)
        if batch_idx % 20 < 10:
            discriminator_loss.backward()
        #discriminator_loss.backward()
        optimizerD.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGen. Loss: {:.6f}, Discr. Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), generator_loss.item(), discriminator_loss.item()))
            #print("discriminator output on fake data : ", gen_disc_output)
            #print("discriminator output on real data : ", true_discriminator_out)
            print(tensor_to_string(gen_output[0].detach().numpy(), voc_list=voc_list))
            print(tensor_to_string(target[0].detach().numpy(), voc_list=voc_list))
            if dry_run:
                break

def train_Gen_supervised(Gen, Disc, device, train_loader, optimizerG, optimizerD,
          epoch, log_interval, dry_run, string_len, batch_size, voc_list):

    loss = nn.BCELoss() #nn.MSELoss()
    Gen.train
    Disc.train
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(target).to(device)
        #print(data.size())

        # zero the gradients on each iteration
        optimizerG.zero_grad()

        #Generated output
        gen_output = Gen(data)
        #print(gen_output.size(), target.size())

        gen_output = nn.Softmax(dim=2)(gen_output)

        #print(gen_disc_output.size(), true_labels.size())
        generator_loss = loss(gen_output, target)
        generator_loss.backward()
        optimizerG.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGen. Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), generator_loss.item(), ))
            print(tensor_to_string(gen_output[0].detach().numpy(), voc_list=voc_list))
            print(tensor_to_string(target[0].detach().numpy(), voc_list=voc_list))

            if dry_run:
                break

def train_Disc(Gen, Disc, device, train_loader, optimizerG, optimizerD,
          epoch, log_interval, dry_run, string_len, batch_size):

    loss = nn.CrossEntropyLoss()#nn.MSELoss()

    Gen.train
    Disc.train

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(target).to(device)
        #print(data.size())

        # zero the gradients on each iteration
        optimizerG.zero_grad()

        #Generated output
        gen_output = Gen(data)
        #print("gen outpur shape : ",gen_output.size())

        #get real data
        true_labels, true_data = get_rand_strings_from_lexicon(string_len=string_len, batch_size=batch_size)
        true_tensors = [string_to_tensor(x, string_len)
                     for x in true_data]
        true_tensors = torch.stack(true_tensors, dim=0)

        #true_labels = torch.tensor(true_labels, device=device).float()
        true_labels = torch.empty(size=(batch_size, 2),device=device).float()
        for i in range(batch_size):
            true_labels[i] = torch.tensor([1, 0])
        #print(true_labels.size())


        #get false data
        false_data = get_rand_strings(string_len=string_len, batch_size=batch_size)
        false_tensors = [string_to_tensor(x, string_len)
                     for x in false_data]
        false_tensors = torch.stack(false_tensors)
        false_labels = torch.empty(size=(batch_size, 2),device=device).float()
        for i in range(batch_size):
            false_labels[i] = torch.tensor([0, 1])
        #print(false_labels.size())



        optimizerD.zero_grad()
        # Train the discriminator on the true data

        true_discriminator_out = Disc(true_tensors)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)
        true_discriminator_loss.backward()

        # Train the discriminator on false data
        false_discriminator_out = Disc(false_tensors)
        false_discriminator_loss = loss(false_discriminator_out, false_labels)
        false_discriminator_loss.backward()

        # Train the discriminator on the generated data
        # add .detach() here think about this
        #generator_discriminator_out = Disc(gen_output.detach())
        #generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        discriminator_loss = (true_discriminator_loss + false_discriminator_loss) / 4
        #discriminator_loss.backward()
        optimizerD.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGen. Loss: {:.6f}, Discr. Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), discriminator_loss.item(), discriminator_loss.item()))
            print("true data : ", true_data[0])
            print("fake data : ", false_data[0])
            print("discriminator output on fake data : ", false_discriminator_out[:4])
            print("discriminator output on real data : ", true_discriminator_out[:4])
            if dry_run:
                break
