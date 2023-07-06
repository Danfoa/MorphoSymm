# Morphological Symmetries (MorphoSymm) in Locomoting Dynamical Systems


Welcome to the Morphological Symmetries (MorphoSymm) repository! Here, you will find a comprehensive set of tools for the identification, study, and exploitation of morphological symmetries in locomoting dynamical systems. These symmetries are commonly found in a wide range of systems, including legged, swimming, and flying animals, robotic systems, and animated characters. As such, this repository aims to provide valuable resources for researchers and practitioners in the fields of Robotics, Computer Graphics, and Computational Biology.

This repository holds the code for the paper: [On discrete symmetries of robotic systems: A data-driven and group-theoretic analysis](https://scholar.google.it/scholar?q=on+discrete+symmetries+of+robotic+systems:+a+data-driven+and+group-theoretic+analysis&hl=en&as_sdt=0&as_vis=1&oi=scholart).
Accepted to *Robotics Science and Systems 2023 (RSS 2023)*. For reproducing the experiments of the paper, please see the master branch.

#### Contents:
- [Installation](#installation)
- [Library of symmetric dynamical systems](#library-of-symmetric-dynamical-systems)
- [Tutorial](#tutorial)
    - [Loading symmetric dynamical systems](#loading-symmetric-dynamical-systems)
    - [Exploiting Morphological Symmetries](#exploiting-morphological-symmetries)
    - [Equivariant Neural Networks](#equivariant-neural-networks)
- [Citation](#citation)
- [Contributing](#contributing)

## Installation:
Simply clone the repository and install it through pip:
```bash
git clone https://github.com/Danfoa/MorphoSymm.git
cd MorphoSymm
pip install -e .
```
## Library of symmetric dynamical systems
The following is a non-exhaustive and expanding list of dynamical systems with Discrete Morphological Symmetries. Each example can be
reproduced in a 3D interactive environment running:
```python
python morpho_symm.robot_symmetry_visualization.py robot=<robot> gui=True 
```
This script functions as an introductory tutorial showing how we define the representations of Discrete Morphological Symmetries in order to perform symmetry transformations on the robot state, and proprioceptive and exteroceptive measurements.
### $\mathcal{G}=\mathcal{C}_2$: Reflection Symmetry
|                                    Cassie                                    |                                                    Atlas   	                                                     |                           Bolt   	                           |                                   Baxter 	                                   |   
|:----------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|       ![cassie](docs/static/animations/cassie-C2-symmetries_anim_static.gif)       | 	 ![atlas](https://user-images.githubusercontent.com/8356912/200183197-94242c57-bd9d-41cb-8a0b-509dceef5cb9.gif) | ![bolt](docs/static/animations/bolt-C2-symmetries_anim_static.gif) |       ![baxter](docs/static/animations/baxter-C2-symmetries_anim_static.gif)       | 	        
| [Solo](https://open-dynamic-robot-initiative.github.io/)  	                	 |                                                    **A1**   	                                                    |                          **HyQ**  	                          |                                Mini-Cheetah	                                 |   
|         ![solo](docs/static/animations/solo-C2-symmetries_anim_static.gif)         |                             ![a1](docs/static/animations/a1-C2-symmetries_anim_static.gif)                             |  ![hyq](docs/static/animations/hyq-C2-symmetries_anim_static.gif)  | ![mini-cheetah](docs/static/animations/mini_cheetah-C2-symmetries_anim_static.gif) |
|                       **Anymal-C** 	                	                        |                                                 **Anymal-B**   	                                                 |                          **B1**  	                           |                                     Go1	                                     |   
|     ![anymal_c](docs/static/animations/anymal_c-C2-symmetries_anim_static.gif)     |                       ![anymal_b](docs/static/animations/anymal_b-C2-symmetries_anim_static.gif)                       |   ![b1](docs/static/animations/b1-C2-symmetries_anim_static.gif)   |          ![go1](docs/static/animations/go1-C2-symmetries_anim_static.gif)          |
|                         **UR-3**  	                	                         |                                                   **UR5**   	                                                    |                         **UR10**  	                          |                                  KUKA-iiwa	                                  |   
|          ![ur3](docs/static/animations/ur3-C2-symmetries_anim_static.gif)          |                            ![ur5](docs/static/animations/ur5-C2-symmetries_anim_static.gif)                            | ![ur10](docs/static/animations/ur10-C2-symmetries_anim_static.gif) |         ![iiwa](docs/static/animations/iiwa-C2-symmetries_anim_static.gif)         |

### $\mathcal{G}=\mathcal{C}_n$: Symmetric Systems with Cyclic Group Symmetries
|       [Trifinger](https://sites.google.com/view/trifinger/home-page)-C3 	       |   
|:-------------------------------------------------------------------------------:|
| 	![trifinger-edu](docs/static/animations/trifinger_edu-C3-symmetries_anim_static.gif) | 	        

### $\mathcal{G}=\mathcal{K}_4$: Klein-Four Symmetry
| [Solo](https://open-dynamic-robot-initiative.github.io/)|                                          HyQ 	                                          |                                        Mini-Cheetah                                        |                                 Anymal-C                                 |                                 Anymal-B                                 |
|:---------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:------------------------------------------------------------------------:|
|                      	![Solo-K4](docs/static/animations/solo-Klein4-symmetries_anim_static.gif)                       | 	 ![hyq](docs/static/animations/hyq-Klein4-symmetries_anim_static.gif) | 	      ![Mini-Cheetah-K4](docs/static/animations/mini_cheetah-Klein4-symmetries_anim_static.gif) | ![anymal_c](docs/static/animations/anymal_c-Klein4-symmetries_anim_static.gif) | ![anymal_c](docs/static/animations/anymal_b-Klein4-symmetries_anim_static.gif) |

### $\mathcal{G}=\mathcal{C}_2\times\mathcal{C}_2\times\mathcal{C}_2$: Regular cube symmetry
|                              [Solo](https://open-dynamic-robot-initiative.github.io/) 	                              |                                         Mini-Cheetah                                          |
|:--------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
|                     	![solo-c2xc2xc2](docs/static/animations/solo-C2xC2xC2-symmetries_anim_static.gif)                     | 	   ![bolt](docs/static/animations/mini_cheetah-C2xC2xC2-symmetries_anim_static.gif) |      

### Addition of new dynamical systems to the library.

This repository aims at becoming a central tool in the exploitation of Morphological Symmetries in Robotics, Computer Graphics and Computational Biology.
Therefore, here we summarize the present and future efforts to enlarge the library of dynamical systems used in each of these fields.

## Tutorial
### Loading symmetric dynamical systems
Each symmetric dynamical system has a configuration file in the folder `morpho_symm/cfg/supervised/robot`. To load one
of these systems, simply use the function `load_symmetric_system` as follows:
```python
from morpho_symm.utils.robot_utils import load_symmetric_system
from hydra import compose, initialize

robot_name = 'solo'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
initialize(config_path="morpho_symm/cfg/supervised/robot", version_base='1.3')
robot_cfg = compose(config_name=f"{robot_name}.yaml")
# Load robot instance and its symmetry group
robot, G = load_symmetric_system(robot_cfg=robot_cfg)
```

The function returns:
- `robot` an instance of the class [`PinBulletWrapper`](https://github.com/Danfoa/MorphoSymm/blob/devel/morpho_symm/robots/PinBulletWrapper.py) (utility class bridging [`pybullet`](https://pybullet.org/wordpress/) and [`pinocchio`](https://github.com/stack-of-tasks/pinocchio)).
-  `G`: the symmetry group of the system of instance [`escnn.group.Group`](https://quva-lab.github.io/escnn/api/escnn.group.html#group)
#### Getting and resetting the state of the system

The system state is defined as $`(\mathbf{q}, \mathbf{v}) \;|\; \mathbf{q} \in \mathrm{Q}, \;\mathbf{q} \in $T_{q}\mathrm{Q}`$, being $\mathrm{Q}$ the space of generalized position coordinates, and $T_{q}\mathrm{Q}$ the space of generalized velocity coordinates. Recall from the [paper convention](https://arxiv.org/abs/2302.10433) that the state configuration can be separated into base configuration and joint-space configuration $`\mathrm{Q} := \mathrm{E}_d \times \mathrm{Q}_J`$, such that $`
\mathbf{q} :=
\begin{bsmallmatrix}
\mathbf{X}_B \\ \mathbf{q}_{js}     
\end{bsmallmatrix}
\begin{smallmatrix}
\in \; \mathbb{E}_d \\ \in \; \mathrm{Q}_J
\end{smallmatrix}
`$. Where, $\mathrm{E}\_d$ is the Euclidean space in which the system evolves, and $`\mathrm{Q}_J`$ is the joint-space position coordinates. To access these quantities in code we do:
```python 
# Get the state of the system
q, v = robot.get_state()  #  q ∈ Q, v ∈ TqQ
# Get the robot's base configuration XB ∈ Ed as a homogenous transformation matrix.
XB = robot.get_base_configuration()
# Get joint space position and velocity coordinates  (q_js, v_js) | q_js ∈ QJ, dq_js ∈ TqQJ
q_js, v_js = robot.get_joint_space_state()
```

### Exploiting Morphological Symmetries

 > _This section shows how to get the group representations required for doing *data augmentation* and for the construction of *equivariant neural networks*_. 
  
The system's symmetry group instance `G` contains the group representations required to transform most proprioceptive and exteroceptive measurements (e.g., joint positions/velocities/accelerations, joint forces/torques, contact locations & forces, linear and angular velocities, terrain heightmaps, depthmaps, etc). These are:


 -  $`\rho_{\mathbb{E}_d}: \mathcal{G} \rightarrow \mathbb{E}(d)`$: Representation mapping symmetry actions to elements of the Euclidean group $\mathbb{E}(d)$. Essentially homogenous transformation matrices describing a rotation/reflection and translation of space (Euclidean isometry) in $d$ dimensions.
 -  $`\rho_{\mathrm{Q}_J}: \mathcal{G} \rightarrow \mathcal{GL}(\mathrm{Q}_J)`$ and $`\rho_{T_q\mathrm{Q}_J}: \mathcal{G} \rightarrow \mathcal{GL}(T_q\mathrm{Q}_J)`$: Representations mapping symmetry actions to transformation matrices of joint-space position $`\mathrm{Q}_J`$ and velocity $`T_{q}\mathrm{Q}_J`$ coordinates. 
 -  $`\rho_{\mathrm{O}_d}: \mathcal{G} \rightarrow \mathcal{\mathrm{O}_d}`$: Representation mapping symmetry actions to elements of the Orthogonal group $\mathrm{O}(d)$. Essentially rotation and reflection matrices in $d$ dimensions.
 -  $`\rho_{reg}: \mathcal{G} \rightarrow \mathbb{R}^{|\mathcal{G}|}`$: The group regular representation.
 -  $`\hat{\rho}_{i}: \mathcal{G} \rightarrow \mathcal{GL}(|\hat{\rho}_{i}|)`$: Each of the group irreducible representations.

In practice, products and additions of these representations are enough to obtain the representations of any proprioceptive and exteroceptive measurement. For instance, we can use these representations to transform elements of:

- $\mathrm{E}\_d$: The Euclidean space (of $d$ dimensions) in which the system evolves.
  
  The representation $`\rho_{\mathbb{E}_d}`$ can be used to transform:
    - The system base configuration $\mathbf{X}\_B$. If you want to obtain the set of symmetric base configurations $`
      \{{g \cdot \mathbf{X}_B:= \rho_{\mathbb{E}_d}(g) \; \mathbf{X}_B  \;  \rho_{\mathbb{E}_d}(g)^{-1} \;|\; \forall\; g \in \mathcal{G}}\}`$ [(1)](https://arxiv.org/abs/2302.1043), you can do it with:

```python 
        rep_Ed = G.representations['Ed']  # rep_Ed(g) is a homogenous transformation matrix ∈ R^(d+1)x(d+1) 
        # The orbit of the base configuration XB is a map from group elements g ∈ G to base configurations g·XB ∈ Ed
        orbit_X_B = {g: rep_Ed(g) @ XB @ rep_Ed(g).T for g in G.elements()} 
```
-
    - Points in $\mathbb{R}^d$. These can represent contact locations, object/body positions, etc. To obtain the set of symmetric points you can do it with:
```python
        r = np.random.rand(3)   # Example point in Ed, assuming d=3.
        r_hom = np.concatenate((r, np.ones(1)))  # Use homogenous coordinates to represent a point in Ed
        # The orbit of the point is a map from group elements g ∈ G to the set of symmetric points g·r ∈ R^d
        orbit_r = {g: (rep_Ed(g) @ r_hom)[:3] for g in G.elements}
```

- $`\mathrm{Q}_J`$ and $`T_{q}\mathrm{Q}_J`$: The spaces of joint-space position $`\mathrm{Q}_J`$ and velocity $`T_{q}\mathrm{Q}_J`$ generalized coordinates.

  To transform joint-space states $`
  (\mathbf{q}_{js}, \mathbf{v}_{js}) \;|\; \mathbf{q}_{js} \in \mathrm{Q}_J, \;\mathbf{v}_{js} \in T_{q}\mathrm{Q}_J
  `$ we use the representations $`\rho_{\mathrm{Q}_J}`$ and $`\rho_{T_q\mathrm{Q}_J}`$. For instance, for a given joint-space configuration $` (\mathbf{q}_{js}, \mathbf{v}_{js})`$, the set of symmetric joint-space configurations (orbit) is given by $`
  \{
  (\rho_{\mathrm{Q}_J}(g) \; \mathbf{q}_{js}, \;\rho_{T_q\mathrm{Q}_J}(g) \;\mathbf{v}_{js}) \; | \; \forall \; g \in \mathcal{G}
  \}
  `$. Equivalently, in code we can do:
```python
    rep_QJ = G.representations['Q_js']     
    rep_TqJ = G.representations['TqQ_js']
    # Get joint space position and velocity coordinates  (q_js, v_js) | q_js ∈ QJ, dq_js ∈ TqQJ
    q_js, v_js = robot.get_joint_space_state()
    # The joint-space state orbit is a map from group elements g ∈ G to joint-space states (g·q_js, g·v_js)  
    orbit_js_state = {g: (rep_QJ(g) @ q_js, rep_TqJ(g) @ v_js) for g in G.elements}
```
- Vectors, Pseudo-vectors in $\mathrm{E}\_d$.

   Vector measurements can represent linear velocities, forces, linear accelerations, etc. While [pseudo-vectors](https://en.wikipedia.org/wiki/Pseudovector#:~:text=In%20physics%20and%20mathematics%2C%20a,of%20the%20space%20is%20changed) (or axial-vectors) can represent angular velocities, angular accelerations, etc. To obtain symmetric measurements we transform vectors with $`
  \rho_{\mathrm{O}_d}`$. Likewise, to obtain symmetric pseudo-vectors we use $`\rho_{\mathrm{O}_{d,pseudo}}(g) := |\rho_{\mathrm{O}_d}(g)| \rho_{\mathrm{O}_d}(g) \; | \; g \in \mathcal{G}`$. Equivalently, in code we can do:
```python
    rep_Od = G.representations['Od'] # rep_Od(g) is an orthogonal matrix ∈ R^dxd
    rep_Od_pseudo = G.representations['Od_pseudo'] 
    
    v = np.random.rand(3)  # Example vector in Ed, assuming d=3. E.g. linear velocity of the base frame.
    w = np.random.rand(3)  # Example pseudo-vector in Ed, assuming d=3. E.g. angular velocity of the base frame.
    # The orbit of the vector is a map from group elements g ∈ G to the set of symmetric vectors g·v ∈ R^d
    orbit_v = {g: rep_Od(g) @ v for g in G.elements}
    # The orbit of the pseudo-vector is a map from group elements g ∈ G to the set of symmetric pseudo-vectors g·w ∈ R^d
    orbit_w = {g: rep_Od_pseudo(g) @ w for g in G.elements}
```

> As an example you can check the script [robot_symmetry_visualization.py](https://github.com/Danfoa/MorphoSymm/blob/devel/morpho_symm/robot_symmetry_visualization.py), where we use the symmetry representations to generate the animations of all robot in the library (with the same script).

### Equivariant Neural Networks

 > _In this section we briefly show how to construct G-equivariant multi-layer perceptron E-MLP architectures. Future tutorials will cover G-equivariant CNNs and GNNs._

Let's consider the example from [(1)](https://arxiv.org/abs/2302.1043) of approximating the Center of Mass (CoM) momentum from the joint-space state measurements. That is we want to use a neural network to approximate the function $`
\mathbf{y} = f(\mathbf{x}) = f(\mathbf{q}_{js}, \mathbf{v}_{js})
`$ for a robot evolving in 3 dimensions, say the robot `solo`. Defining $`\mathbf{y} := [\mathbf{l}, \mathbf{k}]^T \subseteq \mathbb{R}^6`$ as the CoM momentum linear $`\mathbf{l} \in \mathbb{R}^3`$ and angular $`\mathbf{k} \in \mathbb{R}^3`$ momentum, and $`
\mathbf{x} = (\mathbf{q}_{js}, \mathbf{v}_{js}) \;|\; \mathbf{q}_{js} \in \mathrm{Q}_J, \;\mathbf{v}_{js} \in T_{q}\mathrm{Q}_J
`$ as the joint-space position and velocity generalized coordinates.

For this example, you can build an equivariant MLP as follows: 
```python
import escnn
from escnn.nn import FieldType
from hydra import compose, initialize

from morpho_symm.nn.EMLP import EMLP
from morpho_symm.utils.robot_utils import load_symmetric_system

# Load robot instance and its symmetry group
initialize(config_path="morpho_symm/cfg/supervised/robot", version_base='1.3')
robot_name = 'solo'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
robot_cfg = compose(config_name=f"{robot_name}.yaml")
robot, G = load_symmetric_system(robot_cfg=robot_cfg)

# We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
gspace = escnn.gspaces.no_base_space(G)
# Get the relevant group representations.
rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
rep_O3 = G.representations["Od"]  # Used to transform the linear momentum l ∈ R3
rep_O3_pseudo = G.representations["Od_pseudo"]  # Used to transform the angular momentum k ∈ R3

# Define the input and output FieldTypes using the representations of each geometric object.
# Representation of x := [q, v] ∈ Q_js x TqQ_js      =>    ρ_X_js(g) := ρ_Q_js(g) ⊕ ρ_TqQ_js(g)  | g ∈ G
in_field_type = FieldType(gspace, [rep_QJ, rep_TqQJ])
# Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
out_field_type = FieldType(gspace, [rep_O3, rep_O3_pseudo])

# Construct the equivariant MLP
model = EMLP(in_type=in_field_type,
             out_type=out_field_type,
             num_layers=5,              # Input layer + 3 hidden layers + output/head layer
             num_hidden_units=128,      # Number of hidden units per layer
             activation=escnn.nn.ReLU,  # Activarions must be `EquivariantModules` instances
             with_bias=True             # Use bias in the linear layers
             )

print(f"Here is your equivariant MLP \n {model}")
```
____________________________________
In summary, to construct a G-equivariant architecture you need to:
1. Identify the types of geometric objects in your input and output spaces.
2. Identify the representations of each geometric object.
3. Define the input and output `FieldType` instances using the representations of each geometric object.

## How to cite us?
If you find this repository or the [paper](https://scholar.google.it/scholar?q=on+discrete+symmetries+of+robotic+systems:+a+data-driven+and+group-theoretic+analysis&hl=en&as_sdt=0&as_vis=1&oi=scholart) useful, please cite us as:
```
@article{ordonez2023dms_discrete_morphological_symmetries,
  title={On discrete symmetries of robotics systems: A group-theoretic and data-driven analysis},
  author={Ordonez-Apraez, Daniel and Martin, Mario and Agudo, Antonio and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2302.10433},
  year={2023}
}
```

## Contributing

If you have any doubts or ideas, create an issue or contact us. We are happy to help and collaborate.

In case you want to contribute, thanks for being that awesome, and please contact us to see how can we assist you.

#### Robotics
The repository focuses on robotics and uses the URDF (Unified Robot Description Format) to integrate new systems.
It utilizes the [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) package to simplify the integration of new URDF descriptions and their usage in
third-party robotics packages. This package provides a convenient interface for loading URDF files into GUI
visualization tools, robot dynamics packages (such as Pinocchio), and physics simulators. To add a new robotic system to our library
1. The system URDF must be contributed to robot_descriptions.py.
2. The corresponding robot configuration file should be added to `cfg/supervised/robot/` describing the system' symmetry group and joint-space representation generators, should also be added.

In summary, we support:

- [x] Loading of URDF files in `pybullet` and `pinocchio` through `robot_descriptions.py`
- [x] Visualization of robot Discrete Morphological Symmetries in `pybullet`. Other physics simulators and visualization tools will come soon.
- [x] Utility functions to define symmetry representations of proprioceptive and exteroceptive measurements.
- [x] Construction of equivariant neural networks processing proprioceptive and exteroceptive measurements, using the `escnn` library.

#### Computer Graphics

The field of computer graphics does not widely employs URDF descriptions for the definition of dynamical systems. Although covering different description standards is within the goal of this repository,
for now, our main objective is:

- [ ] Integration of [STAR](https://star.is.tue.mpg.de/) model in the library, to automatically process sequence of data and obtain symmetric sequences.
  By defining the sagittal symmetry of all the model parameters. This will enable the use of DMSs in all applications of human motion prediction, shape reconstruction, etc.
  If you are interested in contributing to this effort, please contact us.
- [ ] Integration of Motion Capture (MoCap) data formats. Including `.fbx`, `.bvh`, and `.c3d`.

#### Computational Biology

For computational biology and bio-mechanics, we believe the most relevant format to provide support for is:
- [ ] Coordinate 3D files `.c3d` format. 
