import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group): #self.files is defined in MAPS class
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first

    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            # print(f'step_begin = {step_begin}')
            
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        # print(f"result['audio'].shape = {result['audio'].shape}")
        # print(f"result['label'].shape = {result['label'].shape}")
        return result

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='../../MAESTRO/maestro-v2.0.0', groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v2.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        # tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        tsvs = [f.replace('/flac/', '/tsv_version2/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))

class MAPS_v2(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, refresh=False, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        # Getting the paths
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv_version2/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))