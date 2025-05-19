# HST-WAVE
## Requirements
Our code is based on torch=2.2.0, python=3.8, dgl=2.2.1, pytorch lightning=2.2.2. Please make sure you have installed Python, PyTorch and Pytorch Lightning correctly.

## Data
The two datasets could be found in ./data folder, which contains part of all our data and is designed to show the data format.

Check the paper appendix for details of relevant data.

## Model
Our model HST-WAVE is implemented in "model.py".

## Train
You can train and test HST-Wave directly through the following commands. If you need to modify the parameter configuration of the model, please modify in "train_hstwave.py". 
```
python train_hstwave.py --data jh
python train_hstwave.py --data hz
```
Note: In our implementation, the G56 dataset in the paper corresponds to 'hz' and the G60 dataset in the paper corresponds to 'jh'.