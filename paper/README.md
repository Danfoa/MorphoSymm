## Experiments 
The script used to run the two experiments described in the paper is `train_supervised.py` which receives configuration files using `hydra`. 
The default configurations can be found on `cfg/supervised/config.yaml`. 
All experiments were run in an HPC using `slurm`. The slurm jobs are not made public but the following are some examples of how to run the experiments on standard machines

The `training_supervised` script will create an `experiment` directory on the project root folder, where the results of each individual run will be stored along with a tensorboard log file 
holding several metrics logged during training and validation. After training is done, each model variant is evaluated on the test dataset and a log file `test_metrics.csv` is created 
containing all evaluation metrics of the test set. These `test_metrics.csv` files for each of the model variants and individual seeds are used by the scripts in the `paper` directory 
to generate several figures including the ones presented in the paper. 

### CoM Estimation
- Atlas: Example experiment training `mlp` and `emlp` models with `256`, `512`, and `1024` hidden channel neurons, with and without data augmentation.
```
python train_supervised.py --multirun robot=Atlas dataset=com_momentum dataset.samples=500000 dataset.augment=True,False model.num_channels=256,512,1024  model=emlp,mlp dataset.train_ratio=0. model.lr=1.5e-3 model.inv_dims_sca exp_name=com_sample_eff$
```
- Solo with Morphological Symmetry Group C2 and K4
```
python train_supervised.py --multirun robot=Solo,Solo-c2 dataset=com_momentum dataset.augment=True,False dataset.train_ratio=0.70 model=emlp,mlp model.lr=2.4e-3 model.num_channels=64,128,256,512 exp_name=com_sample_eff
```

### Static Friction Regime Contact Detection 
- CNN
```
python train_supervised.py --multirun robot=mini-cheetah-c2 dataset=contact dataset.data_folder=training_splitted dataset.train_ratio=0.85 dataset.augment=True,False exp_name=contact_sample_eff_splitted model=contact_cnn model.lr=1e-4 
```
- ECNN
```
python train_supervised.py --multirun robot=mini-cheetah-c2 dataset=contact dataset.data_folder=training_splitted dataset.train_ratio=0.85 dataset.augment=False exp_name=contact_sample_eff_splitted model=contact_ecnn model.lr=1e-5 
```
