import os


from datetime import datetime
import pickle

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ex = Experiment('train_transcriber')

# logdir = './runs/Test'

@ex.config
def config():
    controller = 'a'
    
    if controller == 'a':
        logdir = './runs/a'
        
    elif controller == 'b':
        logdir = './runs/b'
    ex.observers.append(FileStorageObserver.create(logdir))
@ex.automain
def train(controller):
    print(f'Hello {controller} World')
    