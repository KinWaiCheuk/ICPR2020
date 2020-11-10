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


from model.evaluate_fn import evaluate_wo_velocity
from model import *

import matplotlib.pyplot as plt
ex = Experiment('Evaluation')

# parameters for the network (These parameters works the best)
ds_ksize, ds_stride = (2,2),(2,2)
mode = 'imagewise'
sparsity = 1
log = True # Turn on log magnitude scale spectrograms.

def removing_nnAudio_parameters(state_dict):
    pop_list = []
    for i in state_dict.keys():
        if i.startswith('spectrogram'):
            pop_list.append(i)

    print(f'The following weights will be remove:\n{pop_list}')
    decision = input("Do you want to proceed? [y/n] ")

    while True:
        if decision.lower()=='y':        
            for i in pop_list:
                state_dict.pop(i)
            return state_dict   
        elif decision.lower()=='n':
            return state_dict  

        print(f'Please choose only [y] or [n]')
        decision = input("Do you want to proceed? [y/n] ")  


@ex.config
def config():
    weight_file = 'MAESTRO-CQT-transcriber_only'
    logdir = os.path.join('results', weight_file)
    unpacked_weight_name = weight_file.split('-')
    spec =unpacked_weight_name[1]
    dataset = 'MAPS'
    device = 'cuda:0'
    
    if weight_file.split('-')[-1] == "transcriber_only":
        reconstruction = False
    elif weight_file.split('-')[-1] == "transcriber_reconstructor":
        reconstruction = True
    print(f'reconstruction = {reconstruction}')
#     reconstruction = True
    
    leave_one_out = None
        
@ex.automain
def train(spec, dataset, device, reconstruction, logdir, leave_one_out, weight_file): 
    print_config(ex.current_run)

    # Choosing the dataset to use
    if dataset == 'MAESTRO':
        validation_dataset = MAESTRO(path='../../MAESTRO/', groups=['test'], sequence_length=None, device=device)

    elif dataset == 'MusicNet':
        validation_dataset = MusicNet(groups=['small test'], sequence_length=None, device=device)

    else:
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=None, overlap=True, device=device)
        

    model = Net(ds_ksize,ds_stride, log=log, reconstruction=reconstruction, mode=mode, spec=spec, norm=sparsity)
    model.to(device)
    model_path = os.path.join('trained_weights', weight_file)
    state_dict = torch.load(model_path)
    model.load_my_state_dict(state_dict)

    summary(model)
    
    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_velocity(tqdm(validation_dataset), model, reconstruction=reconstruction,
                                    save_path=os.path.join(logdir,f'./{dataset}_MIDI_results'))

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')

    export_path = os.path.join(logdir, f'{dataset}_result_dict')    
    pickle.dump(metrics, open(export_path, 'wb'))

