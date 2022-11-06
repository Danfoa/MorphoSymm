This repository holds the preliminary code of the submission to ICRL 2023: `On discrete symmetries of robotic systems: A data-driven and group-theoretic analysis`

**The code to this anonymous repository is made available only to Area/Program chairs and reviewers**, not to any user of OpenReview. We kindly ask you **NOT TO SHARE** this link to any third parties
unrelated to the submission of our paper to ICRL. 

The code, dataset, and weights will be made public on a repository hosted on GitHub upon acceptance. 
Expect changes to the code structure and API, as we intend to improve readability/usability and utility. 

----------------------------------------------

# On discrete symmetries of robotic systems: A data-driven and group-theoretic analysis




 ![Morpholoigal symmetry states of the quadruped robot Solo](https://user-images.githubusercontent.com/8356912/191269534-af143f29-1f46-4009-858b-72a63b5c67ac.gif)  
 ![atlas-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/200183197-94242c57-bd9d-41cb-8a0b-509dceef5cb9.gif)
 ![bolt-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/200183086-98d636d7-75b2-4744-b77f-99b3a1ec8e39.gif)  |
 
A 3D interactive version of this animation is intended to be made public for all robots used in this project, functioning as an educational tool to have a clear understanding of the 
morphological symmetries. For the moment a functional script for Solo and its K4 group can be run on `paper/robot_visualization.py`

## Experiments 
The script used to run the two experiments described in the paper is `train_supervised.py` which receives configuration files using `hydra`. 
The default configurations can be found on the `cgf` directory. 
All experiments were run in an HPC using `slurm`. The slurm jobs are not made public but the following are some examples of how to run the experiments on standard machines

The `training_supervised` script will create an `experiment` directory on the project root folder, where the results of each individual run will be stored along with a tensorboard log file 
holding several metrics logged during training and validation. After training is done, each model variant is evaluated on the test dataset and a log file `test_metrics.csv` is created 
containing all evaluation metrics of the test set. These `test_metrics.csv` files for each of the model variants and individual seeds are used by the scripts in the `paper` directory 
to generate several figures including the ones presented in the paper. 

### CoM Estimation
- Atlas: Example experiment training `mlp` and `emlp` models with `256`, `512`, and `1024` hidden channel neurons, with and without data augmentation.
```
python train_supervised.py --multirun dataset=com_momentum dataset.samples=500000 model.num_channels=256,512,1024 exp_name=com_sample_eff robot_name=Atlas model=emlp,mlp dataset.train_ratio=0.70,0.3,0.1 model.lr=1.5e-3 model.inv_dims_sca dataset.augment=True,False$
```
- Solo with Morphological Symmetry Group C2 and K4
```
python train_supervised.py --multirun dataset=com_momentum exp_name=com_sample_eff robot_name=Solo,Solo-c2 model=emlp,mlp dataset.train_ratio=0.70 model.lr=2.4e-3 dataset.augment=True,False model.num_channels=64,128,256,512
```

### Static Friction Regime Contact Detection 
- CNN
```
python train_supervised.py --multirun dataset=contact dataset.data_folder=training_splitted exp_name=contact_sample_eff_splitted robot_name=mini-cheetah model=contact_cnn dataset.train_ratio=0.85 model.lr=1e-4 dataset.augment=True,False
```
- ECNN
```
python train_supervised.py --multirun dataset=contact dataset.data_folder=training_splitted exp_name=contact_sample_eff_splitted robot_name=mini-cheetah model=contact_ecnn dataset.train_ratio=0.85 model.lr=1e-5 dataset.augment=False
```
