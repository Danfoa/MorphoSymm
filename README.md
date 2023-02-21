# On discrete symmetries of robotic systems: A data-driven and group-theoretic analysis


## Example dynamical systems with Discrete Morphological Symmetries 
The following is a non-exhaustive and expanding list of dynamical systems with Discrete Morphological Symmetries. Each example can be
reproduced in a 3D interactive environment running:
```
cd [...]/RobotEquivariantNN
python robot_symmetry_visualization.py robot=<robot> gui=True debug=False
```
There is a large number of real-world robotic systems with DMSs not listed in this library. For now re restrict 
ourselves to the systems in the curated library of URDF configurations offered by `robot_descriptions.py`. Expect pull request to `robot_descriptions.py` to enlarge this list.

### $\mathcal{G}=\mathcal{C}_2$: Reflection Symmetry
|                                                              Solo-C2   	                                                               |                                                                Atlas   	                                                                |                                                                Bolt   	                                                                |                                                                A1 	                                                                |   
|:--------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------:|
| 	![solo-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/203265566-ca07eb13-8b50-4ee1-ada7-6ebc985c4e30.gif)  | 	 ![atlas-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/200183197-94242c57-bd9d-41cb-8a0b-509dceef5cb9.gif) |  ![bolt-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/200183086-98d636d7-75b2-4744-b77f-99b3a1ec8e39.gif)  | ![a1-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/203263932-1258a540-41d9-4b3d-9eb3-b67a840a7f5a.gif) | 	        
 |                                                             **Cassie**   	                                                             |                                                             **Baxter**   	                                                              |                                                               **HyQ**-C2  	                                                               |                                                                ---	                                                                |   
| ![cassie-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/203263954-331759e7-72da-4530-b5f1-a51c328b8ad6.gif) | ![baxter-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/203263946-7252bcd3-e4e5-48a4-842e-906b50df9122.gif)  | ![hyq-c2-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/203263960-ee553b56-f781-40ac-8daa-d7e1c59f10e7.gif) |                                                               ------                                                               |

### $\mathcal{G}=\mathcal{K}_4$: Klein-Four Symmetry
|                                                     Solo   	                                                      |                                                                HyQ   	                                                                | ----   	 | ---- 	 |   
|:-----------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:------:|
| 	![Solo-K4](https://user-images.githubusercontent.com/8356912/191269534-af143f29-1f46-4009-858b-72a63b5c67ac.gif) | 	 ![hyq-symmetries_anim_static](https://user-images.githubusercontent.com/8356912/203263962-3ee004db-f2f9-468c-ba89-04f3cd316c0d.gif) |  -----   | -----  | 	        

## $\mathcal{G}$-Equivariant NN

On the module `nn/EquivariantModules.py` you can find the $\mathcal{G}$-Equivariant Perceptron (`BasisLinear`) and Convolutional (`BasisConv1d`) layers classes. 
These are the equivariant versions of the standard Pytorch `Linear` and `Conv1D` layers. 

The parametrization of the equivariant layers (i.e., class signature) is identical to the unconstrained Pytorch layer classes.
The only additional parameters are the input output representations `rep_in` and `rep_out`. 
Each representation class instance holds the Symmetry Group and the matrix representations of each of the group actions.

Each example of the library of equivariant dynamical systems holds the representations on the Euclidean Space `Ed` and 
on the system join-space `Q_J`, required to construct the representation for any NN input output spaces.

Expect a tutorial for the use of these classes. For now, visit `nn/ContactCNN` and `nn/EMLP` for the classes of the 
NN architectures used in the experiments of the paper.

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
