"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
# from .mel import melspectrogram
from .constants import *
from nnAudio import Spectrogram
from .tcn import TemporalConvNet

melspectrogram = Spectrogram.MelSpectrogram(SAMPLE_RATE, WINDOW_LENGTH, N_MELS, HOP_LENGTH,
                                            fmin=MEL_FMIN, fmax=MEL_FMAX, trainable_mel=True, device=DEFAULT_DEVICE)


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size), #(batch, 640, 768)
            sequence_model(model_size, model_size), #(batch, 640, 768)
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 2, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']
        # offset_label = batch['offset']
        # velocity_label = batch['velocity']
        
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        onset_pred, _, frame_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

class Frames_LSTM(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)


        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    def forward(self, mel):
        frame_pred = self.frame_stack(mel)
        return frame_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        frame_label = batch['frame']
        # offset_label = batch['offset']
        # velocity_label = batch['velocity']
        
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        frame_pred = self(mel)

        predictions = {
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = torch.transpose(x, -1,-2) # swapping (features, seq) to (seq, features)
        x = self.tcn(x)
        x =  torch.transpose(x, -1,-2)
        return self.linear(x)

# TCN_layers = [3*2**x for x in range(9)]
# TCN_layers = [3, 3, 6, 6, 12, 12, 24, 48, 96, 192, 384, 768]
# TCN_layers = [100, 400, 600, 768]

class Onset_Stack(nn.Module):
    def __init__(self, input_features, model_size, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = ConvStack(input_features, model_size)
        self.layer2 = TCN(model_size, output_features, TCN_layers, 7, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

class Combined_Stack(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = TCN(input_features, output_features, TCN_layers, 7, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return x

class OnsetsAndFrames_TCN(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = Onset_Stack(input_features, model_size, output_features, TCN_layers)

        # self.onset_stack = nn.Sequential(
        #     ConvStack(input_features, model_size), #(batch, 640, 768)
        #     sequence_model(model_size, model_size), #(batch, 640, 768)
        #     nn.Linear(model_size, output_features),
        #     nn.Sigmoid()
        # )

        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = Combined_Stack(output_features*2, output_features, TCN_layers)
        # self.combined_stack = nn.Sequential(
        #     sequence_model(output_features * 2, model_size),
        #     nn.Linear(model_size, output_features),
        #     nn.Sigmoid()
        # )

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        onset_pred, _, frame_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses


def reverse_sequence(tensor):
    idx = [i for i in range(tensor.size(1)-1, -1, -1)]
    idx = torch.LongTensor(idx).cuda()
    return tensor.index_select(1, idx)    


class biTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(biTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(2*num_channels[-1], output_size)

    def forward(self, x):
        x = torch.transpose(x, -1,-2) # swapping (features, seq) to (seq, features)
        normal_x = self.tcn(x)
        reversed_x = self.tcn(reverse_sequence(x))
        x = torch.cat((normal_x,reverse_sequence(reversed_x)),1)
        x =  torch.transpose(x, -1,-2)
        return self.linear(x)

class biTCNv2(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(biTCNv2, self).__init__()
        self.tcn_forward = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn_backward = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(2*num_channels[-1], output_size)

    def forward(self, x):
        x = torch.transpose(x, -1,-2) # swapping (features, seq) to (seq, features)
        normal_x = self.tcn_forward(x)
        reversed_x = self.tcn_backward(reverse_sequence(x))
        x = torch.cat((normal_x,reverse_sequence(reversed_x)),1)
        x =  torch.transpose(x, -1,-2)
        return self.linear(x)

class Onset_Stack_bi(nn.Module):
    def __init__(self, input_features, model_size, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = ConvStack(input_features, model_size)
        self.layer2 = biTCN(model_size, output_features, TCN_layers, kernel_size, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

class Onset_Stack_biv2(nn.Module):
    def __init__(self, input_features, model_size, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = ConvStack(input_features, model_size)
        self.layer2 = biTCNv2(model_size, output_features, TCN_layers, kernel_size, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

class Combined_Stack_bi(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = biTCN(input_features, output_features, TCN_layers, kernel_size, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return x

class Combined_Stack_biv2(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = biTCNv2(input_features, output_features, TCN_layers, kernel_size, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return x

class OnsetsAndFrames_biTCN(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16

        self.onset_stack = Onset_Stack_bi(input_features, model_size, output_features, TCN_layers, kernel_size)

        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        # self.combined_stack = Combined_Stack(output_features*2, output_features, TCN_layers, kernel_size)
        # Previously forgot to use bi
        self.combined_stack = Combined_Stack_bi(output_features*2, output_features, TCN_layers, kernel_size)

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        # print(f'mel shape = {mel.shape}')
        onset_pred, _, frame_pred = self(mel)
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

class OnsetsAndFrames_biTCNv2(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16

        self.onset_stack = Onset_Stack_biv2(input_features, model_size, output_features, TCN_layers, kernel_size)

        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        # self.combined_stack = Combined_Stack(output_features*2, output_features, TCN_layers, kernel_size)
        # Previously forgot to use bi
        self.combined_stack = Combined_Stack_biv2(output_features*2, output_features, TCN_layers, kernel_size)

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        # print(f'mel shape = {mel.shape}')
        onset_pred, _, frame_pred = self(mel)
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses



class biTCN_fully(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(biTCN_fully, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        x = torch.transpose(x, -1,-2) # swapping (features, seq) to (seq, features)
        normal_x = self.tcn(x)
        reversed_x = self.tcn(reverse_sequence(x))
        x = torch.cat((normal_x,reverse_sequence(reversed_x)),1) # should also try adding
        x =  torch.transpose(x, -1,-2)
        return x



class Onset_Stack_bi_fully(nn.Module):
    def __init__(self, input_features, model_size, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = ConvStack(input_features, model_size)
        self.layer2 = biTCN_fully(model_size, output_features, TCN_layers, kernel_size, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

class Combined_Stack_bi_fully(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size):
        super().__init__()    

        self.layer1 = biTCN_fully(input_features, output_features, TCN_layers, kernel_size, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return x


class OnsetsAndFrames_biTCN_fully(nn.Module):
    def __init__(self, input_features, output_features, TCN_layers, kernel_size, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16

        self.onset_stack = Onset_Stack_bi_fully(input_features, model_size, output_features, TCN_layers, kernel_size)

        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = Combined_Stack_bi_fully(output_features*2, output_features, TCN_layers, kernel_size)

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        frame_label = batch['frame']


        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        mel = mel.transpose(-1,-2) # swap mel bins with timesteps so that it fits LSTM later # shape (8,640,229)
        onset_pred, _, frame_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            # 'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses