# DA6401-A2

Note: The notebook has been updated gradually as the code developed, the scripts have been generated from the notebook directly

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Notebook Functions](#notebook-functions)
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)

---

## Project Overview
**Objective**: Train a customizable CNN model to classify images from the iNaturalist dataset.  
**Key Features**:
- Flexible architecture with configurable hyperparameters
- Integration with WandB for hyperparameter sweeps
- Dynamic model dimension calculation
- Stratified validation splitting

---

## Installation
1. Clone repository:

git clone https://github.com/yourusername/inaturalist-cnn.git
cd inaturalist-cnn

2. Install requirements

pip install -r requirements.txt


---

## Notebook Functions

### Core Components
| Function/Class              | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `CNN`               | Customizable CNN model with 5 conv blocks + dense layers                    |
| `DataManager`     | Handles data loading, splitting, and preprocessing                         |
| `train_sweep()`             | Main training function for WandB hyperparameter sweeps                      |
| `load_data()`               | Loads and preprocesses iNaturalist dataset from zip                         |
| `create_prediction_grid()`  | Generates visual grid of model predictions                                  |

### Hyperparameter Configuration

sweep_config = {
"method": "bayes",
"metric": {"name": "val_acc", "goal": "maximize"},
"parameters": { # Contains 12 tunable parameters
"filter_base":,
"conv_activation": ["relu", "gelu", "silu", "mish"],
... (full config in notebook)
}
}

---

## Usage

### Running the Notebook
1. Add dataset to `/data` folder:
2. Configure WandB:
3. Start sweep:

---

## Troubleshooting

### Common Issues

#### 1. WandB Authentication Errors

###### Solution

Relogin with correct API key

wandb login --relogin

#### 2. CUDA Out of Memory

##### Solution

Reduce batch size

#### 3. Missing Data Folders

##### Solution

Ensure that the directory architecture matches as required by the script

---

## Requirements

- torch==2.0.1
- torchvision==0.15.2
- pytorch-lightning==2.0.0
- wandb==0.15.5
- numpy==1.24.3
- matplotlib==3.7.1


