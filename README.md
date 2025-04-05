# Three-Layer-MLP-NN
Three Layer Multi-Layer Perceptron Neural Network

## Installation
```bash
git clone https://github.com/Michael-Yi-Rong/Three-Layer-MLP-NN.git
```

## Set conda environment
```bash
conda create -n drivex python=3.7
conda activate MLP
```

## Install requirements
```bash
pip install -r requirements.txt
```

## File structure
```bash
- data/
- models/
- plots/
  ├── heatmap/
  ├── hist/
  └── train/
- grid_search.py
- model.py
- plotting.py
- train.py
- test.py
- utils.py
- read_npz.py
- requirements.txt
- README.md
```

## Train the model
```bash
python ./train.py
```

## Train the model with grid search
### Change the parameter combinations in grid_search.py and run it, the model will be saved in "./models/"
```bash
python ./grid_search.py
```

## Test the model
```bash
python ./test.py
```

## Plotting the model
### Plotting W1 b1 W2 b2 with heatmaps and hists, the plots will be saved in "./plots/"
```bash
python ./plotting.py
```

## Already trained model
### You can download this from google drive to "./models" and use it, enjoy it!
```bash
https://drive.google.com/drive/folders/1MHdcqiQxPysy11eNPTZDZQx8h78bJeYr?usp=sharing
```
