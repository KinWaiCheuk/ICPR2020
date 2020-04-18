import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnAudio import Spectrogram
from .constants import *

batchNorm_momentum = 0.1
num_instruments = 1

def normalize(x):
    size = x.shape
    
    x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
    x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]

    x_max = x_max.unsqueeze(1) # Make it broadcastable
    x_min = x_min.unsqueeze(1) # Make it broadcastable 
    
    return (x-x_min)/(x_max-x_min)

# melspectrogram = Spectrogram.MelSpectrogram(SAMPLE_RATE, WINDOW_LENGTH, N_MELS, HOP_LENGTH,
                                            #  fmin=MEL_FMIN, fmax=MEL_FMAX, trainable_mel=True, device=DEFAULT_DEVICE)
r=2
N_MELS = 88*r
melspectrogram = Spectrogram.CQT1992v2(SAMPLE_RATE, HOP_LENGTH, device=DEFAULT_DEVICE, n_bins=N_MELS,bins_per_octave=12*r)
class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride) 
        else: 
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 

    def forward(self, x, size=None, isLast=None, skip=None):
        # print(f'x.shape={x.shape}')
        # print(f'target shape = {size}')
        x = self.us(x,output_size=size)
        if not isLast: x = torch.cat((x, skip), 1) 
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))
        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Encoder, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

    def forward(self, x):
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
       
        c1=self.conv1(x3) 
        c2=self.conv2(x2) 
        c3=self.conv3(x1) 
        return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]
        

class Decoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Decoder, self).__init__()
        # self.d_block1 = d_block(320,64,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block4 = d_block(16,num_instruments,True,(3,3),(1,1),ds_ksize, ds_stride)
            

            
    def forward(self, x, s, c=[None,None,None,None]):
        x = self.d_block1(x,s[3],False,c[0])
        x = self.d_block2(x,s[2],False,c[1])
        x = self.d_block3(x,s[1],False,c[2])

        reconsturction = torch.sigmoid(self.d_block4(x,s[0],True,c[3]))
       
        return reconsturction

    
class Decoder_Pianoroll(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Decoder_Pianoroll, self).__init__()
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block4 = d_block(16,num_instruments,True,(3,3),(1,1),ds_ksize, ds_stride)
        
        self.lstm = nn.LSTM(N_MELS, N_MELS, batch_first=True, bidirectional=True)    
        self.linear = nn.Linear(N_MELS*2, 88)

        # self.linear = nn.Linear(N_MELS, 88)

            
    def forward(self, x, s, c=[None,None,None,None]):
        x = self.d_block1(x,s[3],False,c[0])
        x = self.d_block2(x,s[2],False,c[1])
        x = self.d_block3(x,s[1],False,c[2])
        x = self.d_block4(x,s[0],True,c[3])
        x, h = self.lstm(x.squeeze(1)) # remove the channel dim
        x = self.linear(x) # Use the full LSTM output
        x = torch.sigmoid(x)
        return x

class Encoder_Pianoroll(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Encoder_Pianoroll, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

        self.lstm = nn.LSTM(88, N_MELS, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(N_MELS*2, N_MELS)

    def forward(self, x):
        x, h = self.lstm(x)
        x = self.linear(x)
        x1,idx1,s1 = self.block1(x.unsqueeze(1))
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
       
        c1=self.conv1(x3) 
        c2=self.conv2(x2) 
        c3=self.conv3(x1) 
        return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]


class Net(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True):
        super(Net, self).__init__()
        self.encoder = Encoder(ds_ksize, ds_stride)
        self.decoder = Decoder(ds_ksize, ds_stride)
        self.pianoroll_decoder = Decoder_Pianoroll(ds_ksize, ds_stride)
        self.pianoroll_encoder = Encoder_Pianoroll(ds_ksize, ds_stride)
        self.log = log

    def forward(self, x):
        vec1,s1,c1 = self.encoder(x)
        pianoroll = self.pianoroll_decoder(vec1,s1,c1)

        vec2,s2,c2 = self.pianoroll_encoder(pianoroll)

        # z = torch.cat((vec1, vec2), 1) # cat along the channel axis
        z = vec2
        reconstruction = self.decoder(vec2,s2,c2) # predict roll
        
        return z, reconstruction, pianoroll


    def run_on_batch(self, batch, semi_supervised=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        if self.log:
            mel = torch.log(mel + 1e-5)
            mel = normalize(mel)
        # print(f'mel shape = {mel.shape}')

        vec, reconstrut, pianoroll = self(mel.view(mel.size(0), 1, mel.size(1), mel.size(2)))
        predictions = {
            # 'onset': onset_pred.reshape(*onset_label.shape),
            # # 'offset': offset_pred.reshape(*offset_label.shape),
            # 'frame': frame_pred.reshape(*frame_label.shape),
            # # 'velocity': velocity_pred.reshape(*velocity_label.shape)
            'onset': pianoroll,
            'frame': pianoroll,
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'reconstruction': reconstrut
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)

            
        }
        if semi_supervised:
            losses = {
                # 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

                # 'loss/onset': onset_label,

                    'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), mel.detach())
                    }
        else:
            losses = {
                # 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

                # 'loss/onset': onset_label,

                    'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), mel.detach()),
                    'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label)
                    }

        return predictions, losses, mel

class Net_double_transcripters(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True):
        super(Net_double_transcripters, self).__init__()
        self.encoder = Encoder(ds_ksize, ds_stride)
        self.decoder = Decoder(ds_ksize, ds_stride)
        self.pianoroll_decoder = Decoder_Pianoroll(ds_ksize, ds_stride)
        self.pianoroll_encoder = Encoder_Pianoroll(ds_ksize, ds_stride)
        self.log = log

    def forward(self, x):
        vec,s,c = self.encoder(x)
        pianoroll = self.pianoroll_decoder(vec,s,c)

        vec,s,c = self.pianoroll_encoder(pianoroll)

        reconstruction = self.decoder(vec,s,c) # predict roll

        vec,s,c = self.encoder(reconstruction)
        pianoroll2 = self.pianoroll_decoder(vec,s,c)
        
        return vec, reconstruction, pianoroll, pianoroll2


    def run_on_batch(self, batch, semi_supervised=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        if self.log:
            mel = torch.log(mel + 1e-5)
            mel = normalize(mel)
        # print(f'mel shape = {mel.shape}')

        vec, reconstrut, pianoroll, pianoroll2 = self(mel.view(mel.size(0), 1, mel.size(1), mel.size(2)))
        predictions = {
            # 'onset': onset_pred.reshape(*onset_label.shape),
            # # 'offset': offset_pred.reshape(*offset_label.shape),
            # 'frame': frame_pred.reshape(*frame_label.shape),
            # # 'velocity': velocity_pred.reshape(*velocity_label.shape)
            'onset': pianoroll,
            'frame': pianoroll,
            'frame2':pianoroll2,
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'reconstruction': reconstrut
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)

            
        }
        if semi_supervised:
            losses = {
                # 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

                # 'loss/onset': onset_label,

                    'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), mel.detach())
                    }
        else:
            losses = {
                # 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

                # 'loss/onset': onset_label,

                    'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), mel.detach()),
                    'loss/transcription': F.binary_cross_entropy(predictions['frame'].squeeze(1), frame_label),
                    'loss/transcription2': F.binary_cross_entropy(predictions['frame2'].squeeze(1), frame_label)
                    }

        return predictions, losses, mel


class LSTM_AE(nn.Module):
    def __init__(self, ds_ksize, ds_stride):
        super(LSTM_AE, self).__init__()
        self.encoder1 = nn.LSTM(N_MELS, N_MELS, batch_first=True, bidirectional=True)
        self.encoder2 = nn.Linear(N_MELS*2, 88)

        self.decoder1 = nn.LSTM(88, 88, batch_first=True, bidirectional=True)
        self.decoder2 = nn.Linear(88*2, N_MELS)
    def forward(self, x):

        x, _ = self.encoder1(x)
        x = self.encoder2(x)
        pianoroll = torch.sigmoid(x) # This is the pianoroll

        x, _ = self.decoder1(pianoroll)
        x = self.decoder2(x)
        
        return pianoroll, x


    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        # print(f'mel shape = {mel.shape}')
        
        pianoroll, reconstrut = self(mel.view(mel.size(0), mel.size(1), mel.size(2)))
        predictions = {
            'onset': pianoroll,
            'frame': pianoroll,
            'reconstruction': reconstrut
        }

        losses = {
            'loss/reconstruction': F.mse_loss(reconstrut.squeeze(1), mel.detach()),
            'loss/transcription': F.binary_cross_entropy(predictions['frame'], frame_label)
        }

        return predictions, losses, mel

class Net_wo_reconstruct(nn.Module):
    def __init__(self):
        super(Net_wo_reconstruct, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.pianoroll_decoder = Decoder_Pianoroll()
#         self.pianoroll_decoder = Decoder_Pianoroll(x)

    def forward(self, x):
        vec,s,c = self.encoder(x)
        reconstruction = self.decoder(vec,s,c) # predict roll
        pianoroll = self.pianoroll_decoder(vec,s,c)
        return vec, reconstruction, pianoroll


    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        # print(f'mel shape = {mel.shape}')
        vec, reconstrut, pianoroll = self(mel.view(mel.size(0), 1, mel.size(1), mel.size(2)))
        predictions = {
            # 'onset': onset_pred.reshape(*onset_label.shape),
            # # 'offset': offset_pred.reshape(*offset_label.shape),
            # 'frame': frame_pred.reshape(*frame_label.shape),
            # # 'velocity': velocity_pred.reshape(*velocity_label.shape)
            'onset': pianoroll,
            'frame': pianoroll,
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'reconstruction': reconstrut
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)

            
        }

        losses = {
            # 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

            # 'loss/onset': onset_label,
            # 'loss/reconstruction': F.mse_loss(reconstrut.squeeze(0), mel),
            'loss/transcription': F.binary_cross_entropy(predictions['frame'], frame_label)
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        }

        return predictions, losses