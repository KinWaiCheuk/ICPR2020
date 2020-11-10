import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnAudio import Spectrogram
from .constants import *
from .Unet_blocks import *
import sys

batchNorm_momentum = 0.1
num_instruments = 1



class Normalization():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""
    def __init__(self, mode='framewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                output = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                output[torch.isnan(output)]=0 # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                return (x-x_min)/(x_max-x_min)
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def transform(self, x):
        return self.normalize(x)

class Net(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, reconstruction=True, mode='framewise', spec='CQT', norm=1, device='cpu'):
        super(Net, self).__init__()
        global N_BINS # using the N_BINS parameter from constant.py
        
        # Selecting the type of spectrogram to use
        if spec == 'CQT':
            r=2
            N_BINS = 88*r
            self.spectrogram = Spectrogram.CQT1992v2(sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                      n_bins=N_BINS, fmin=27.5, 
                                                      bins_per_octave=12*r, trainable=False)            
        elif spec == 'Mel':
            self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS,
                                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                          trainable_mel=False, trainable_STFT=False)
        else:
            print(f'Please select a correct spectrogram')                

        self.log = log
        self.normalize = Normalization(mode)
        self.norm = norm            
        self.reconstruction = reconstruction            
            
        self.Unet1_encoder = Encoder(ds_ksize, ds_stride)
        self.Unet1_decoder = Decoder(ds_ksize, ds_stride)
        self.lstm1 = nn.LSTM(N_BINS, N_BINS, batch_first=True, bidirectional=True)    
        self.linear1 = nn.Linear(N_BINS*2, 88)        

        if reconstruction==True:
            self.Unet2_encoder = Encoder(ds_ksize, ds_stride)
            self.Unet2_decoder = Decoder(ds_ksize, ds_stride)   
            self.lstm2 = nn.LSTM(88, N_BINS, batch_first=True, bidirectional=True)
            self.linear2 = nn.Linear(N_BINS*2, N_BINS)             
               
    def forward(self, x):

        # U-net 1
        x,s,c = self.Unet1_encoder(x)
        feat1 = self.Unet1_decoder(x,s,c)
        x, h = self.lstm1(feat1.squeeze(1)) # remove the channel dim
        pianoroll = torch.sigmoid(self.linear1(x)) # Use the full LSTM output

        if self.reconstruction:        
            # U-net 2
            x, h = self.lstm2(pianoroll)
            feat2= torch.sigmoid(self.linear2(x)) # ToDo, remove the sigmoid activation and see if we get a better result
            x,s,c = self.Unet2_encoder(feat2.unsqueeze(1))
            reconstruction = self.Unet2_decoder(x,s,c) # predict roll

            # Applying U-net 1 to the reconstructed spectrograms
            x,s,c = self.Unet1_encoder(reconstruction)
            feat1b = self.Unet1_decoder(x,s,c)
            x, h = self.lstm1(feat1b.squeeze(1)) # remove the channel dim
            pianoroll2 = torch.sigmoid(self.linear1(x)) # Use the full LSTM output

            return feat1, feat2, feat1b, reconstruction, pianoroll, pianoroll2
        else:
            return feat1, pianoroll



    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)
            
        # Converting audio to spectrograms
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        
        # log compression
        if self.log:
            spec = torch.log(spec + 1e-5)
            
        # Normalizing spectrograms
        spec = self.normalize.transform(spec)
        
        # swap spec bins with timesteps so that it fits LSTM later 
        spec = spec.transpose(-1,-2) # shape (8,640,229)

        
        if self.reconstruction:
            feat1, feat2, feat1b, reconstrut, pianoroll, pianoroll2 = self(spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))
            predictions = {
                'onset': pianoroll,
                'frame': pianoroll,
                'frame2':pianoroll2,
                'onset2':pianoroll2,
                'reconstruction': reconstrut,
                'feat1': feat1,
                'feat2': feat2,
                'feat1b': feat1b
                }
            losses = {
                    'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), spec.detach()),
                    'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                    'loss/transcription2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label)
                    }

            return predictions, losses, spec

        else:
            feat1, pianoroll = self(spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))
            predictions = {
                'onset': pianoroll,
                'frame': pianoroll,
                'feat1': feat1,
                }
            losses = {
                    'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label)
                    }

            return predictions, losses, spec            
        
    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class Net_only_transcripters_shared(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, mode='framewise', spec='CQT', norm=1):
        super(Net_only_transcripters_shared, self).__init__()
        # global SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, 'cpu', N_MELS, MEL_FMIN, MEL_FMAX
        global N_BINS

        if spec == 'CQT':
            r=2
            N_BINS = 88*r
            self.spectrogram = Spectrogram.CQT1992v2(SAMPLE_RATE, HOP_LENGTH, device='cpu', n_bins=N_BINS,bins_per_octave=12*r, trainable=False)            
        elif spec == 'Mel':
            self.spectrogram = Spectrogram.MelSpectrogram(SAMPLE_RATE, WINDOW_LENGTH, N_BINS, HOP_LENGTH,fmin=MEL_FMIN, fmax=MEL_FMAX, trainable_mel=False, trainable_STFT=False, device='cpu')
        else:
            print(f'Please select a correct spectrogram')

        self.encoder = Encoder(ds_ksize, ds_stride)
        self.pianoroll_decoder = Decoder_Pianoroll(ds_ksize, ds_stride)
        self.log = log
        self.normalize = Normalization(mode)
        self.norm = norm

        self.lstm = nn.LSTM(N_BINS, N_BINS, batch_first=True, bidirectional=True)    
        self.linear = nn.Linear(N_BINS*2, 88)

    def forward(self, x):

        x,s,c = self.encoder(x)
        feat = torch.sigmoid(self.pianoroll_decoder(x,s,c))

        x, h = self.lstm(feat.squeeze(1)) # remove the channel dim
        x = self.linear(x) # Use the full LSTM output
        pianoroll = torch.sigmoid(x)        

        return feat, pianoroll


    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        torch.save(audio_label, 'audio')
        if frame_label.dim() == 2:
            frame_label = frame_label.unsqueeze(0)

        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)

        feat, pianoroll = self(spec.view(spec.size(0), 1, spec.size(1), spec.size(2)))
        predictions = {
            'onset': pianoroll,
            'frame': pianoroll,
            'feat': feat
            }
        losses = {
                'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label)
                }


        return predictions, losses, spec

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

