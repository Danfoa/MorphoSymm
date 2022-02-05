import math
import os
import pathlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from emlp.reps import Vector
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from com_momentum import get_robot_params
from groups.SymmetricGroups import C2
from nn.EquivariantModel import EquivariantModel
from nn.EquivariantModules import EMLP, MLP, BasisLinear, EBlock
from nn.datasets import COMMomentum
from utils.utils import slugify


class Identity(torch.nn.Module):
    def forward(self, x):
        return x

def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True, ax=None):
    columns = len(val_dict)
    if ax is None:
        fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))
    else:
        assert len(ax) == columns, "Ups"
        fig = ax[0].get_figure()

    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(
            val_dict[key],
            ax=key_ax,
            color=color,
            bins=50,
            legend=True,
            stat=stat,
            kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
        )  # Only plot kde if there is variance
        hidden_dim_str = (
            r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape) > 1 else ""
        )
        key_ax.set_title(f"{key} {hidden_dim_str}")
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1
    fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
    return ax


def visualize_weight_distribution(model, color="C0", ax=None):
    weights = {}

    # Plot resultant W
    for layer_index, layer in enumerate(model.net):
        if isinstance(layer, torch.nn.Linear):
            for name, param in layer.named_parameters():
                if name.endswith(".bias"):
                    continue
                s = name.split('.')
                key_name = f"Layer{layer_index}.{s[-1]}"
                weights[key_name] = param.detach().view(-1).cpu().numpy()
        elif isinstance(layer, EBlock):
            W = layer.linear.W.view(-1).detach().cpu().numpy()
            basis_coeff = layer.linear.basis_coeff.view(-1).detach().cpu().numpy()
            weights[f"Layer{layer_index}.c"] = basis_coeff
            weights[f"Layer{layer_index}.W"] = W
        elif isinstance(layer, BasisLinear):
            W = layer.W.view(-1).detach().cpu().numpy()
            basis_coeff = layer.basis_coeff.view(-1).detach().cpu().numpy()
            weights[f"Layer{layer_index}.c"] = basis_coeff
            weights[f"Layer{layer_index}.W"] = W

    # Plotting
    ax = plot_dists(weights, color=color, xlabel="Weight vals", ax=ax)
    # fig.suptitle("Weight distribution", fontsize=14, y=0.97)
    # plt.show()
    # plt.close()
    return ax[0].get_figure(), ax


def visualize_gradients(model, data_loader, color="C0", print_variance=False, grad_keyword="weight", loss_fn=F.mse_loss,
                        ax=None):
    """
    Args:
        net: Object of class BaseNetwork
        color: Color in which we want to visualize the histogram (for easier separation of activation functions)
    """
    model.eval()
    x, y = next(iter(data_loader))

    # Pass one batch through the network, and calculate the gradients for the weights
    model.zero_grad()
    preds = model(x)
    loss = loss_fn(preds, y)
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    params = dict(model.named_parameters())
    grads = {
        name: params.grad.view(-1).cpu().clone().numpy()
        for name, params in model.named_parameters()
        if params.grad is not None
    }
    model.zero_grad()

    # Plotting
    ax = plot_dists(grads, color=color, xlabel="Grad magnitude", ax=ax)
    # fig.suptitle("Gradient distribution", fontsize=14, y=0.97)
    # plt.show()
    # plt.close()

    if print_variance:
        for key in sorted(grads.keys()):
            print(f"{key} - Variance: {np.var(grads[key])}")

    return ax[0].get_figure(), ax

def visualize_activations(model, data_loader, color="C0", print_variance=False, ax=None):
    model.eval()
    x, y = next(iter(data_loader))

    # Pass one batch through the network, and calculate the gradients for the weights
    feats = x.view(x.shape[0], -1)
    activations = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.net):
            if isinstance(layer, EBlock):
                feats = layer.linear(feats)
                activations[f"Layer {layer_index}"] = feats.view(-1).detach().cpu().numpy()
                feats = layer.activation(feats)
            else:
                feats = layer(feats)
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, BasisLinear) :
                activations[f"Layer {layer_index}"] = feats.view(-1).detach().cpu().numpy()

    # Plotting
    ax = plot_dists(activations, color=color, stat="density", xlabel="Activation vals", ax=ax)
    # fig.suptitle("Activation distribution", fontsize=14, y=0.97)
    # plt.show()
    # plt.close()

    if print_variance:
        for key in sorted(activations.keys()):
            print(f"{key} - Variance: {np.var(activations[key])}")

    return ax[0].get_figure(), ax

def weights_initialization(module,
                           weights_initializer=torch.nn.init.kaiming_uniform,
                           bias_initializer=torch.nn.init.zeros_):
    # TODO: Place default initializer
    # TODO: Compute initialization considering weight sharing distribution
    # For now use Glorot initialization, must check later:
    if isinstance(module, EBlock):
        basis_layer = module.linear
        gain = 1 # np.sqrt(2) if isinstance(module.activation, torch.nn.SiLU) else 1

        fan_in, fan_out = basis_layer.W.shape
        if weights_initializer == torch.nn.init.xavier_uniform:
            # Xavier cannot find the in out dimensions because the tensor is not 2D
            # TODO: Check this
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(basis_layer.basis_coeff, -bound, bound)
        elif weights_initializer == torch.nn.init.kaiming_uniform:
            std = gain / math.sqrt(fan_out)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                return basis_layer.basis_coeff.uniform_(-bound, bound)
        else:
            weights_initializer(basis_layer.basis_coeff)
        if basis_layer.with_bias:
            bias_initializer(basis_layer.bias_basis_coeff)


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    cfg.seed = 10
    seed_everything(seed=cfg.seed)
    # Avoid repeating to compute basis at each experiment.
    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    robot_name = "bolt"
    robot, G_in, GC = get_robot_params(robot_name)

    # Parameters
    activations = [Identity, torch.nn.SiLU, torch.nn.ReLU]
    activations_names = ["Identity", "Swish", "ReLU"]
    initializers = [torch.nn.init.kaiming_uniform, torch.nn.init.xavier_uniform, torch.nn.init.normal]

    for activation, act_name in zip(activations, activations_names):
        fig_ac, ax_ac = plt.subplots(nrows=len(initializers), ncols=cfg.num_layers + 1,
                                     figsize=((cfg.num_layers + 1) * 3, len(initializers) * 3))
        fig_gr, ax_gr = plt.subplots(nrows=len(initializers), ncols=cfg.num_layers + 1,
                                     figsize=((cfg.num_layers + 1) * 3, len(initializers) * 3))
        fig_w, ax_w = plt.subplots(nrows=len(initializers), ncols=(cfg.num_layers + 1) * 2,
                                   figsize=((cfg.num_layers + 1) * 2 * 3, len(initializers) * 3))

        for i, initializer in enumerate(initializers):
            # Define output group for linear momentum
            G_out = GC.canonical_group(3)
            activation = torch.nn.SiLU #Identity
            model_type = cfg.model_type.lower()
            if model_type == "emlp":
                network = EMLP(rep_in=Vector(G_in), rep_out=Vector(G_out),
                               group=C2, num_layers=cfg.num_layers, ch=cfg.num_channels,
                               with_bias=False, activation=activation,
                               cache_dir=cache_dir).to(dtype=torch.float32)
                network.save_cache_file()
                network.apply(lambda x: weights_initialization(x, initializer))
                weights_kw = "basis"
            elif model_type == 'mlp':
                network = MLP(d_in=G_in.d, d_out=G_out.d, num_layers=cfg.num_layers,
                              ch=cfg.num_channels).to(dtype=torch.float32)
                weights_kw = "weight"
            else:
                raise NotImplementedError(model_type)

            dataset = COMMomentum(robot, cfg.dataset.num_samples, angular_momentum=cfg.dataset.angular_momentum)
            data_loader = DataLoader(dataset, batch_size=cfg.batch_size)

            visualize_activations(network, data_loader=data_loader, ax=ax_ac[i,:], color=f"C{i}")
            visualize_gradients(network, data_loader=data_loader, grad_keyword=weights_kw, ax=ax_gr[i,:], color=f"C{i}")
            visualize_weight_distribution(network, ax=ax_w[i,:], color=f"C{i}")

        title = act_name
        initializer_names = [f"{slugify(init.__name__)}" for init in initializers]
        fig_ac.suptitle(f"[{title}]Activation distribution \n{initializer_names}", fontsize=12, y=0.97)
        fig_ac.savefig(root_path.joinpath(f"media/{title}_l{cfg.num_layers}_act.png"), dpi=120)
        fig_ac.show()
        fig_gr.suptitle(f"[{title}] Gradient distribution \n{initializer_names}", fontsize=12, y=0.97)
        fig_gr.savefig(root_path.joinpath(f"media/{title}_l{cfg.num_layers}_grad.png"), dpi=120)
        fig_gr.show()
        fig_w.suptitle(f"[{title}] W distribution \n{initializer_names}", fontsize=12, y=0.97)
        fig_w.savefig(root_path.joinpath(f"media/{title}_l{cfg.num_layers}_W.png"), dpi=120)
        fig_w.show()

if __name__ == "__main__":
    main()
