import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from evaluate import evaluate, evaluate_wo_velocity # These two lines requires GPU
from onsets_and_frames import *
from onsets_and_frames.transcriber import OnsetsAndFrames_TCN, OnsetsAndFrames_biTCN
ex = Experiment('train_transcriber')




@ex.config
def config():
    logdir = 'runs/TCN_bi7-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 5e-4
    learning_rate_decay_steps = 300
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    # TCN_layers = [6*4**x for x in range(4)]
    # TCN_layers = [3*2**x for x in range(8)]
    TCN_layers = [600, 500, 400, 300, 200, 100]
    if resume_iteration is None:
        model = OnsetsAndFrames_biTCN(N_MELS, MAX_MIDI - MIN_MIDI + 1, TCN_layers, 2, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    epoches = 2000
    total_batch = len(loader.dataset)
    for ep in range(1, epoches):
        model.train()
        total_loss = 0
        batch_idx = 0
        # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
        for batch in loader:
            predictions, losses = model.run_on_batch(batch)

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

        if (ep+1)%10 == 0 and ep+1 > 20:
            model.eval()
            with torch.no_grad():
                for key, values in evaluate_wo_velocity(validation_dataset, model).items():
                    if key.startswith('metric/'):
                        _, category, name = key.split('/')
                        print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                        if ('precision' in name or 'recall' in name or 'f1' in name) and 'chroma' not in name:
                            writer.add_scalar(key, np.mean(values), global_step=ep+1)

        if (ep+1)%50 == 0:
            torch.save(model, os.path.join(logdir, f'model-{ep+1}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep+1)

        # if i % validation_interval == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         for key, value in evaluate(validation_dataset, model).items():
        #             writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
        #     model.train()

        # if i % checkpoint_interval == 0:
        #     torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
        #     torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
