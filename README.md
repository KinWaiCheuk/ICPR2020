# ICPR2020: The Effect of Spectrogram Reconstructions on Automatic Music Transcription
This repository is for the [paper](https://arxiv.org/abs/2010.09969) accepted in ICPR2020. This code uses the PyTorch version of Onsets and Frames written by [jongwook](https://github.com/jongwook/onsets-and-frames) as a template.

## Requirement
* torch == 1.6.0
* torchvision == 0.7.0
* tensorboard == 2.2.0
* numpy == 1.19.1
* matplotlib == 3.0.2
* sacred == 0.8.1
* nnAudio == 0.2.0


## Training the model
The python script can be run using using the sacred syntax `with`.
```python
python train.py with train_on=<arg> spec=<arg> reconstruction=<arg> device=<arg>
```

* `train_on`: the dataset to be trained on. Either `MAPS` or `MAESTRO` or `MusicNet`
* `spec`: the input spectrogram type. Either `Mel` or `CQT`.
* `reconstruction`: to include the reconstruction loss or not. Either `True` or `False`
* `device`: the device to be trained on. Either `cpu` or `cuda:0`

## Evaluating the model and exporting the midi files

```python
python evaluate.py with weight_file=<arg> reconstruction=<arg> device=<arg>
```

* `weight_file`: The weight files should be located inside the `trained_weight` folder
* `dataset`: which dataset to evaluate on, can be either `MAPS` or `MAESTRO` or `MusicNet`.
* `device`: the device to be trained on. Either `cpu` or `cuda:0`

The transcripted midi files, accuracy reports are saved inside the `results` folder.