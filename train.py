import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
import utils
import itertools
import progressbar
import numpy as np
import os
from sklearn.model_selection import train_test_split

import cv2
from torch.utils.data import Dataset
from path import Path
import scipy
import subprocess
import librosa
from skimage import transform as tf
import shutil
import dlib
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from torch.utils.data import Dataset, DataLoader
from librosa.core import load
from pydub import AudioSegment
from pydub.utils import mediainfo
from data_loader import VideoDataset

import torchvision.utils as vutils
from torchvision.utils import save_image

from torch.optim import RMSprop,Adam,SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import imageio.core.util

import time
import soundfile as sf

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='./log', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='rmsprop', help='optimizer to train with')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='gridcorpus', help='dataset to train with')
parser.add_argument('--n_future', type=int, default=75, help='number of frames to predict during training')
parser.add_argument('--nsample', type=int, default=3, help='number of diverse outputs to generate')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

parser.add_argument('--audio_feat_len', default=0.2)
parser.add_argument('--aud_enc_dim', default=128)
parser.add_argument('--audio_rate', default=50000)
parser.add_argument('--n_frame', type=int, default=75, help='number of frames in a video')
parser.add_argument('--q_levels', type=int, default=256, help='number of bins in quantization of audio samples')

parser.add_argument("--recon_level",default=3,action="store",type=int,dest="recon_level")
parser.add_argument("--decay_lr",default=0.75,action="store",type=float,dest="decay_lr")
parser.add_argument("--lambda_mse",default=1e-6,action="store",type=float,dest="lambda_mse")

opt = parser.parse_args()

if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-batch_size=%d-lr=%.6f-z_dim=%d-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.batch_size, opt.lr, opt.z_dim, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
    encoder = saved_model['encoder']
    decoder = saved_model['decoder']
    audio_encoder = saved_model['audio_encoder']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

import models.rnn_audio as aud_enc
if opt.model_dir != '':
    audio_encoder = saved_model['audio_encoder']
else:
    audio_encoder = aud_enc.RNN()
    audio_encoder.apply(utils.init_weights)

frame_predictor_optimizer = RMSprop(params=frame_predictor.parameters(),lr=opt.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
lr_frame_predictor = ExponentialLR(frame_predictor_optimizer, gamma=opt.decay_lr)
posterior_optimizer = RMSprop(params=posterior.parameters(),lr=opt.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
lr_posterior = ExponentialLR(posterior_optimizer, gamma=opt.decay_lr)
prior_optimizer = RMSprop(params=prior.parameters(),lr=opt.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
lr_prior = ExponentialLR(prior_optimizer, gamma=opt.decay_lr)
encoder_optimizer = RMSprop(params=encoder.parameters(),lr=opt.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
lr_encoder = ExponentialLR(encoder_optimizer, gamma=opt.decay_lr)
decoder_optimizer = RMSprop(params=decoder.parameters(),lr=opt.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
lr_decoder = ExponentialLR(decoder_optimizer, gamma=opt.decay_lr)
audio_encoder_optimizer = RMSprop(params=audio_encoder.parameters(),lr=opt.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
lr_audio_encoder = ExponentialLR(audio_encoder_optimizer, gamma=opt.decay_lr)

model_parameters = filter(lambda p: p.requires_grad, frame_predictor.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of frame_predictor trainable parameters: ", params)
model_parameters = filter(lambda p: p.requires_grad, posterior.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of posterior trainable parameters: ", params)
model_parameters = filter(lambda p: p.requires_grad, prior.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of prior trainable parameters: ", params)
model_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of encoder trainable parameters: ", params)
model_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of decoder trainable parameters: ", params)
model_parameters = filter(lambda p: p.requires_grad, audio_encoder.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of audio_encoder trainable parameters: ", params)

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
audio_encoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data = VideoDataset(train = True, image_width=opt.image_width)
test_data = VideoDataset(train = False, image_width=opt.image_width)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)

print(len(train_loader.dataset))
print(len(test_loader.dataset))

# --------- Initialize hidden states ------------------------------------
def init_hidden(batch_size, hidden_size, n_layers):
        hidden = []
        for i in range(n_layers):
            hidden.append((Variable(torch.zeros(batch_size, hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, hidden_size).cuda())))
        return hidden

# --------- plotting funtions ------------------------------------
def plot0(x, y, audio_seq, epoch):
    for n in range(opt.nsample):
        frame_predictor_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.predictor_rnn_layers)
        posterior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.posterior_rnn_layers)
        prior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.prior_rnn_layers)

        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        _, skip = encoder(x_in)

        for i in range(1, opt.n_future):
            h, _ = encoder(x_in)
            h_aud = audio_encoder(y[i])
            z, mu, _, prior_hidden = prior(h_aud, prior_hidden)
            z_t = z
            h, frame_predictor_hidden = frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
            x_in = decoder([h, skip])
            gen_seq.append(x_in)

        random_index = torch.LongTensor(1).random_(0, opt.batch_size)

        aud = audio_seq[0][random_index].squeeze(0)
        audio_chunk = aud[4000:6000]
        audio = audio_chunk
        for i in range(1, opt.n_future):
            aud = audio_seq[i][random_index].squeeze(0)
            audio_chunk = aud[4000:6000]
            audio = torch.cat((audio, audio_chunk), 0)
        
        # librosa.output.write_wav('%s/temp.wav' % (opt.log_dir), audio.cpu().numpy(), 50000, norm=True)
        sf.write('%s/temp.wav' % (opt.log_dir), audio.cpu().numpy(), 50000, 'PCM_24')
    
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('%s/temp.avi' % (opt.log_dir), fourcc, 25, (opt.image_width, opt.image_width))
        for i in range(len(gen_seq)):
            frame = gen_seq[i][random_index]
            frame = utils.save_tensors_image1(frame)
            r, g, b = frame[0, :, :], frame[1, :, :], frame[2, :, :]
            R, G, B = np.uint8(255*r.cpu().numpy()), np.uint8(255*g.cpu().numpy()), np.uint8(255*b.cpu().numpy())
            frm = cv2.merge((B, G, R))
            out.write(frm)
        out.release()
    
        subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s/temp.avi -i %s/temp.wav %s/gen/epoch%s_sample0_%s.avi' % (str(opt.log_dir), str(opt.log_dir), str(opt.log_dir), str(epoch), str(n)), shell=True)
        subprocess.call('rm -r %s/temp.avi' % (opt.log_dir), shell=True)
        subprocess.call('rm -r %s/temp.wav' % (opt.log_dir), shell=True)

def plot1(x, y, audio_seq, epoch):
    random_index = torch.LongTensor(1).random_(0, opt.batch_size)
    for n in range(opt.nsample):
        frame_predictor_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.predictor_rnn_layers)
        posterior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.posterior_rnn_layers)
        prior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.prior_rnn_layers)

        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        _, skip = encoder(x_in)

        for i in range(1, opt.n_future):
            h, _ = encoder(x_in)
            h_aud = audio_encoder(y[i])
            z, mu, _, prior_hidden = prior(h_aud, prior_hidden)
            z_t = z
            h, frame_predictor_hidden = frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
            x_in = decoder([h, skip])
            gen_seq.append(x_in)

        aud = audio_seq[0][random_index].squeeze(0)
        audio_chunk = aud[4000:6000]
        audio = audio_chunk
        for i in range(1, opt.n_future):
            aud = audio_seq[i][random_index].squeeze(0)
            audio_chunk = aud[4000:6000]
            audio = torch.cat((audio, audio_chunk), 0)
        
        # librosa.output.write_wav('%s/temp.wav' % (opt.log_dir), audio.cpu().numpy(), 50000, norm=True)
        sf.write('%s/temp.wav' % (opt.log_dir), audio.cpu().numpy(), 50000, 'PCM_24')
    
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('%s/temp.avi' % (opt.log_dir), fourcc, 25, (opt.image_width, opt.image_width))
        for i in range(len(gen_seq)):
            frame = gen_seq[i][random_index]
            frame = utils.save_tensors_image1(frame)
            r, g, b = frame[0, :, :], frame[1, :, :], frame[2, :, :]
            R, G, B = np.uint8(255*r.cpu().numpy()), np.uint8(255*g.cpu().numpy()), np.uint8(255*b.cpu().numpy())
            frm = cv2.merge((B, G, R))
            out.write(frm)
        out.release()
    
        subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s/temp.avi -i %s/temp.wav %s/gen/epoch%s_sample1_%s.avi' % (str(opt.log_dir), str(opt.log_dir), str(opt.log_dir), str(epoch), str(n)), shell=True)
        subprocess.call('rm -r %s/temp.avi' % (opt.log_dir), shell=True)
        subprocess.call('rm -r %s/temp.wav' % (opt.log_dir), shell=True)

def plot_rec(x, epoch):
    frame_predictor_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.predictor_rnn_layers)
    posterior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.posterior_rnn_layers)

    gen_seq = []
    gen_seq.append(x[0])
    
    x_in = x[0]
    _, skip = encoder(x_in)

    for i in range(1, opt.n_future):
        h, _ = encoder(x[i-1])
        h_target, _ = encoder(x[i])
        z_t, mu, logvar, posterior_hidden = posterior(h_target, posterior_hidden)
        h_pred, frame_predictor_hidden = frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
        x_pred = decoder([h_pred, skip])
        gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 1)
    for i in range(nrow):
        row = []
        for t in range(opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

# --------- training funtions ------------------------------------
def train(x, y):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()
    audio_encoder.zero_grad()

    # initialize the hidden state.
    frame_predictor_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.predictor_rnn_layers)
    posterior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.posterior_rnn_layers)
    prior_hidden = init_hidden(opt.batch_size, opt.rnn_size, opt.prior_rnn_layers)

    mse = 0
    kld = 0
    for i in range(1, opt.n_future):
        h_aud = audio_encoder(y[i])  #[10, 128]
        h_target = encoder(x[i])[0]
        if i==1:
            h, skip = encoder(x[i-1])
        else:
            h, _ = encoder(x[i-1])
        z_t, mu, logvar, posterior_hidden = posterior(h_target, posterior_hidden)
        _, mu_p, logvar_p, prior_hidden = prior(h_aud, prior_hidden)
        h_pred, frame_predictor_hidden = frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)
    
    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    audio_encoder_optimizer.step()

    return mse.data.cpu().numpy()/opt.n_future, kld.data.cpu().numpy()/opt.n_future

# --------- training loop ------------------------------------
length = np.int(len(train_loader.dataset)/opt.batch_size)
for epoch in range(opt.epochs):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    audio_encoder.train()

    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(max_value=length).start()
    for i, (audio_seq, mfcc_seq, frame_seq) in enumerate(train_loader):
        progress.update(i+1)
        x = utils.normalize_frame(opt, dtype, frame_seq)
        y = utils.normalize_audio(opt, dtype, mfcc_seq)

        # train frame_predictor 
        mse, kld = train(x, y)
        epoch_mse += mse
        epoch_kld += kld

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/length, epoch_kld/length, epoch*length*opt.batch_size))

    fname = '%s/loss.txt' % (opt.log_dir)
    if epoch==0:
        with open(fname, 'w') as f:
            print('[%02d] mse loss: %.5f | kld loss: %.5f (%d) \n' % (epoch, epoch_mse/length, epoch_kld/length, epoch*length*opt.batch_size), file=f)
    else:
        with open(fname, 'a') as f:
            print('[%02d] mse loss: %.5f | kld loss: %.5f (%d) \n' % (epoch, epoch_mse/length, epoch_kld/length, epoch*length*opt.batch_size), file=f)

    # plot some stuff
    frame_predictor.eval()
    posterior.eval()
    prior.eval()
    #encoder.eval()
    #decoder.eval()
    #audio_encoder.eval()

    for i, (audio_seq, mfcc_seq, frame_seq) in enumerate(test_loader):
        x = utils.normalize_frame(opt, dtype, frame_seq)
        y = utils.normalize_audio(opt, dtype, mfcc_seq)
        audio_seq = utils.normalize_audio(opt, dtype, audio_seq)

        plot_rec(x, epoch)
        plot0(x, y, audio_seq, epoch)
        plot1(x, y, audio_seq, epoch)

        if i==0:
            break

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'audio_encoder': audio_encoder,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)