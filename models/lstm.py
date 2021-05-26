import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import torch.nn.functional as F

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())

    def forward(self, input, hidden):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        return self.output(h_in), hidden

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        z = eps.mul(logvar).add_(mu)

        return z, mu, logvar, hidden

class Image_EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Image_EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, ten, out=False,t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.lrelu(ten)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = self.lrelu(ten)
            return ten

class Image_Discriminator_64(nn.Module):
    def __init__(self, channel_in=3,recon_level=3):
        super(Image_Discriminator_64, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.size = 32
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=256))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512,momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
        )
        self.sigmoid = nn.Sigmoid()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementation
                    scale = 1.0/numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /=numpy.sqrt(3)
                    #nn.init.xavier_normal(m.weight,1)
                    #nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten_orig, ten_predicted, ten_sampled, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                    ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return self.sigmoid(ten)


    def __call__(self, *args, **kwargs):
        return super(Image_Discriminator_64, self).__call__(*args, **kwargs)

class Image_Discriminator_128(nn.Module):
    def __init__(self, channel_in=3,recon_level=4):
        super(Image_Discriminator_128, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.size = 16
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=32))
        self.size = 32
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(Image_EncoderBlock(channel_in=self.size, channel_out=256))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512,momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
        )
        self.sigmoid = nn.Sigmoid()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementation
                    scale = 1.0/numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /=numpy.sqrt(3)
                    #nn.init.xavier_normal(m.weight,1)
                    #nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten_orig, ten_predicted, ten_sampled, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                    ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return self.sigmoid(ten)


    def __call__(self, *args, **kwargs):
        return super(Image_Discriminator_128, self).__call__(*args, **kwargs)
            
class Video_EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Video_EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, ten, out=False,t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.lrelu(ten)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = self.lrelu(ten)
            return ten

class Video_Discriminator_64(nn.Module):
    def __init__(self, channel_in=3,recon_level=3):
        super(Video_Discriminator_64, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        nf = 64
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv3d(channel_in, nf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True)))
        self.conv.append(Video_EncoderBlock(channel_in=nf, channel_out=nf*2))
        self.conv.append(Video_EncoderBlock(channel_in=nf*2, channel_out=nf*4))
        self.conv.append(Video_EncoderBlock(channel_in=nf*4, channel_out=nf*8))
        # final fc to get the score (real or fake)
        self.conv.append(nn.Sequential(
                nn.Conv3d(nf * 8, 1, 4, 1, 0),
                nn.BatchNorm3d(1),
                ))
        self.sigmoid = nn.Sigmoid()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementation
                    scale = 1.0/numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /=numpy.sqrt(3)
                    #nn.init.xavier_normal(m.weight,1)
                    #nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten_orig, ten_predicted, ten_sampled, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            return self.sigmoid(ten)


    def __call__(self, *args, **kwargs):
        return super(Video_Discriminator_64, self).__call__(*args, **kwargs)

class Video_Discriminator_128(nn.Module):
    def __init__(self, channel_in=3,recon_level=4):
        super(Video_Discriminator_128, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        nf = 32
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv3d(channel_in, nf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True)))
        self.conv.append(Video_EncoderBlock(channel_in=nf, channel_out=nf*2))
        self.conv.append(Video_EncoderBlock(channel_in=nf*2, channel_out=nf*4))
        self.conv.append(Video_EncoderBlock(channel_in=nf*4, channel_out=nf*8))
        self.conv.append(Video_EncoderBlock(channel_in=nf*8, channel_out=nf*16))
        # final fc to get the score (real or fake)
        self.conv.append(nn.Sequential(
                nn.Conv3d(nf * 16, 1, 4, 1, 0),
                nn.BatchNorm3d(1),
                ))
        self.sigmoid = nn.Sigmoid()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementation
                    scale = 1.0/numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /=numpy.sqrt(3)
                    #nn.init.xavier_normal(m.weight,1)
                    #nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten_orig, ten_predicted, ten_sampled, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            return self.sigmoid(ten)


    def __call__(self, *args, **kwargs):
        return super(Video_Discriminator_128, self).__call__(*args, **kwargs)