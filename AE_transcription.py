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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from evaluate import evaluate, evaluate_wo_velocity # These two lines requires GPU
from onsets_and_frames import *
from AE_model import *
import matplotlib.pyplot as plt

ex = Experiment('train_transcriber')

def save_dict(obj, name ):
    with open('./runs/config/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('./runs/config/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

ds_ksize, ds_stride = (2,2),(2,2)

@ex.config
def config():
    # logdir = f'runs_AE/CQTbatch32_WNet_narrow_z_dsksize{ds_ksize}_dsstride_{ds_stride}_' + '-' + datetime.now().strftime('%y%m%d-%H%M%S')
    logdir = f'runs_AE/MAESTRO-Test_on_MAPS-'+ datetime.now().strftime('%y%m%d-%H%M%S')
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'

    batch_size = 16
    sequence_length = 327680
    model_complexity = 48
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    ex.observers.append(FileStorageObserver.create(logdir))

    refresh = False


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, refresh):
    print_config(ex.current_run)
    save_dict(config(), os.path.basename(logdir))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        # validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, refresh=refresh)

    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length, refresh=refresh)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, refresh=refresh)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, 4, shuffle=False, drop_last=True)
    batch_visualize = next(iter(valloader)) # Getting one fixed batch for visualization

    if resume_iteration is None:
        model = Net(ds_ksize,ds_stride, log=True)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        trained_dir='trained_MAPS'
        model_path = os.path.join(trained_dir, f'{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # optimizer.load_state_dict(torch.load(os.path.join(trained_dir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    epoches = 2000
    total_batch = len(loader.dataset)
    for ep in range(1, epoches+1):
        model.train()
        total_loss = 0
        batch_idx = 0
        # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
        for batch in loader:
            predictions, losses, _ = model.run_on_batch(batch)

            loss = sum(losses.values())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if clip_gradient_norm:
                clip_grad_norm_(model.parameters(), clip_gradient_norm)
            batch_idx += 1
            print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
                    f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
                    f'\tLoss: {loss.item():.6f}'
                    , end='\r') 
        print(' '*100, end = '\r')            
        print(f'Train Epoch: {ep}\tLoss: {total_loss/len(loader):.6f}')

        if (ep)%10 == 0 and ep > 1:
            model.eval()
            with torch.no_grad():
                for key, values in evaluate_wo_velocity(validation_dataset, model).items():
                    if key.startswith('metric/'):
                        _, category, name = key.split('/')
                        print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                        if ('precision' in name or 'recall' in name or 'f1' in name) and 'chroma' not in name:
                            writer.add_scalar(key, np.mean(values), global_step=ep)

        if (ep)%50 == 0:
            torch.save(model, os.path.join(logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep)

        # Load one batch from validation_dataset
            
        predictions, losses, mel = model.run_on_batch(batch_visualize)


        # torch.save(mel, 'mel_normal.pt')
        # torch.save(batch['audio'], 'audio.pt')

        if ep==1: # Showing the original transcription and spectrograms
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Original', fig , ep)

            fig, axs = plt.subplots(2, 2, figsize=(24,4))
            axs = axs.flat
            for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Label', fig , ep)

        if ep < 11 or (ep%50 == 0):
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(predictions['reconstruction'].cpu().detach().numpy().squeeze(1)):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()
            
            writer.add_figure('images/Reconstruction', fig , ep)   

            fig, axs = plt.subplots(2, 2, figsize=(24,4))
            axs = axs.flat
            for idx, i in enumerate(predictions['frame'].detach().cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Transcription', fig , ep)

        # writer.add_images('images/Reconstruction', predictions['reconstruction'], ep+1, dataformats='NCWH')

        # if i % validation_interval == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         for key, value in evaluate(validation_dataset, model).items():
        #             writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
        #     model.train()

        # if i % checkpoint_interval == 0:
        #     torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
        #     torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
