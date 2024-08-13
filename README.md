# MorphoSymm Contact Experiment Replication

This branch replicates the Contact Estimation experiment outlined in the paper ["On discrete symmetries of robotics systems: A group-theoretic and data-driven analysis"](https://arxiv.org/abs/2302.10433). See [Issue #9](https://github.com/Danfoa/MorphoSymm/issues/9) for more details.

## Installation:
First, install the following library:
```
sudo aptitude install libnccl2
```

Use the `conda_env.yml` file to create the conda environment:
```
conda env create -f conda_env.yml
conda activate rss2023
```

## Import and Edit Submodule
Run the following command:
```
git submodule init
git submodule update --force --recursive --remote
```

Change lines 16 and 17 in "deep_contact_estimator/src/test.py" to the following:
```
from .contact_cnn import *
from ..utils.data_handler import *
```

## Run experiments
You'll need to run the commands below 8 times in order to generate the 8 random runs with different seeds.

CNN & CNN-aug Experiments:
```
python train_supervised.py --multirun dataset=contact dataset.data_folder=training_splitted exp_name=contact_sample_eff_splitted robot_name=mini-cheetah model=contact_cnn dataset.train_ratio=0.85 model.lr=1e-4 dataset.augment=True,False
```

ECNN Experiment:
```
python train_supervised.py --multirun dataset=contact dataset.data_folder=training_splitted exp_name=contact_sample_eff_splitted robot_name=mini-cheetah model=contact_ecnn dataset.train_ratio=0.85 model.lr=1e-5 dataset.augment=False
```

## Generate Figures

Paper Figure 4-Left & Center:

TODO

Paper Figure 4-Right:
```
python paper/contact_final_model_comparison.py
```