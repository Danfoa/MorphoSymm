import math
import os
import pathlib
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from nn.EquivariantModules import EMLP, MLP, BasisLinear, EBlock, LinearBlock
from nn.datasets import COMMomentum
from utils.utils import slugify


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def extract_weight_distribution(model):
    weights, basis_coeff_w, basis = {}, {}, {}

    for layer_index, layer in enumerate(model.net):
        if isinstance(layer, torch.nn.Linear):
            for name, param in layer.named_parameters():
                if name.endswith(".bias"):
                    continue
                key_name = layer_index
                weights[key_name] = param.detach().view(-1).cpu().numpy()
        elif isinstance(layer, LinearBlock):
            W = layer.linear.weight.view(-1).detach().cpu().numpy()
            weights[layer_index] = W
        elif isinstance(layer, EBlock):
            W = layer.linear.weight.view(-1).detach().cpu().numpy()
            weights[layer_index] = W
            basis_coeff = layer.linear.basis_coeff.view(-1).detach().cpu().numpy()
            basis_coeff_w[layer_index] = basis_coeff
            base = torch.sum(layer.linear.basis, dim=-1).view(-1).detach().cpu().numpy()
            basis[layer_index] = base
        elif isinstance(layer, BasisLinear):
            W = layer.weight.view(-1).detach().cpu().numpy()
            weights[layer_index] = W
            basis_coeff = layer.basis_coeff.view(-1).detach().cpu().numpy()
            basis_coeff_w[layer_index] = basis_coeff
            base = torch.sum(layer.basis, dim=-1).view(-1).detach().cpu().numpy()
            basis[layer_index] = base

    df_weights = pd.concat([pd.DataFrame.from_dict({k: v}) for k, v in weights.items()], axis=1)
    df_weights = df_weights.melt(id_vars=None, value_vars=df_weights.columns, var_name="Layer", value_name="Param_Value").dropna()
    df_weights["Param"] = "W"
    if len(basis_coeff_w) > 0:
        df_basis_coeff = pd.concat([pd.DataFrame.from_dict({k: v}) for k, v in basis_coeff_w.items()], axis=1)
        df_basis_coeff = df_basis_coeff.melt(id_vars=None, value_vars=df_basis_coeff.columns,
                                             var_name="Layer", value_name="Param_Value").dropna()
        df_basis_coeff["Param"] = "c"
        df_weights = pd.concat((df_weights, df_basis_coeff), axis=0)

        df_basis = pd.concat([pd.DataFrame.from_dict({k: v}) for k, v in basis.items()], axis=1)
        df_basis = df_basis.melt(id_vars=None, value_vars=df_basis.columns, var_name="Layer",
                                       value_name="Param_Value").dropna()
        df_basis["Param"] = "basis"
        df_weights = pd.concat((df_weights, df_basis), axis=0)

    return df_weights


def extract_gradients(model, data_loader, loss_fn=F.mse_loss):
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

    # Store gradients of layer weights in Equivariant module
    for layer_index, layer in enumerate(model.net):
        if isinstance(layer, EBlock):
            layer.linear._weight.retain_grad()
        elif isinstance(layer, BasisLinear):
            layer._weight.retain_grad()


    loss = loss_fn(preds, y)
    loss.backward()

    layer_grads, layer_basis_coeff_grads = {}, {}
    for layer_index, layer in enumerate(model.net):
        if isinstance(layer, torch.nn.Linear):
            for name, param in layer.named_parameters():
                if name.endswith(".bias"):
                    continue
                layer_grads[layer_index] = param.grad.view(-1).cpu().clone().numpy()
        elif isinstance(layer, LinearBlock):
            grad = layer.linear.weight.grad.view(-1).detach().cpu().clone().numpy()
            layer_grads[layer_index] = grad
        elif isinstance(layer, EBlock):
            grad = layer.linear._weight.grad.view(-1).detach().cpu().clone().numpy()
            basis_coeff_grad = layer.linear.basis_coeff.grad.view(-1).detach().cpu().clone().numpy()
            layer_basis_coeff_grads[layer_index] = basis_coeff_grad
            layer_grads[layer_index] = grad
        elif isinstance(layer, BasisLinear):
            grad = layer._weight.grad.view(-1).detach().cpu().clone().numpy()
            basis_coeff_grad = layer.basis_coeff.grad.view(-1).detach().cpu().clone().numpy()
            layer_basis_coeff_grads[layer_index] = basis_coeff_grad
            layer_grads[layer_index] = grad

    df_grads = pd.concat([pd.DataFrame.from_dict({k: v}) for k, v in layer_grads.items()], axis=1)
    df_grads = df_grads.melt(id_vars=None, value_vars=df_grads.columns, var_name="Layer", value_name="Grad").dropna()
    df_grads["Param"] = "W"
    if len(layer_basis_coeff_grads) > 0:
        df_basis_coeff_grads = pd.concat([pd.DataFrame.from_dict({k: v}) for k, v in layer_basis_coeff_grads.items()], axis=1)
        df_basis_coeff_grads = df_basis_coeff_grads.melt(id_vars=None, value_vars=df_basis_coeff_grads.columns,
                                                         var_name="Layer", value_name="Grad").dropna()
        df_basis_coeff_grads["Param"] = "c"
        df_grads = pd.concat((df_grads, df_basis_coeff_grads), axis=0)

    model.zero_grad()
    return df_grads


def extract_activations(model, data_loader):
    model.eval()
    x, y = next(iter(data_loader))

    # Pass one batch through the network, and calculate the gradients for the weights
    feats = x.view(x.shape[0], -1)
    activations = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.net):
            print(f"-{layer_index}: {layer}")
            if isinstance(layer, EBlock):
                feats = layer.linear(feats)
                activations[layer_index] = feats.view(-1).detach().cpu().numpy()
                feats = layer.activation(feats)
            elif isinstance(layer, LinearBlock):
                feats = layer.linear(feats)
                activations[layer_index] = feats.view(-1).detach().cpu().numpy()
                feats = layer.activation(feats)
            elif isinstance(layer, torch.nn.Linear) or isinstance(layer, BasisLinear):
                feats = layer(feats)
                activations[layer_index] = feats.view(-1).detach().cpu().numpy()
            else:
                raise NotImplementedError(type(layer))
            print(f"\t Activations: {feats.shape}")

    df = pd.concat([pd.DataFrame.from_dict({k:v}) for k,v in activations.items()], axis=1)
    df = df.melt(id_vars=None, value_vars=df.columns, var_name="Layer", value_name="Activation").dropna()
    return df


def weights_initialization(module, activation,
                           weights_initializer=torch.nn.init.kaiming_uniform,
                           bias_initializer=torch.nn.init.zeros_):
    # TODO: Place default initializer
    # TODO: Compute initialization considering weight sharing distribution
    # For now use Glorot initialization, must check later:
    if isinstance(module, EBlock):
        basis_layer = module.linear
        gain = 1  # np.sqrt(2) if isinstance(module.activation, torch.nn.SiLU) else 1

        fan_in, fan_out = basis_layer.weight.shape
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
    elif isinstance(module, torch.nn.Linear):
        try:
            weights_initializer(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity=activation))
        except Exception as e:
            try:
                weights_initializer(module.weight, nonlinearity=activation)
            except Exception as e:
                warnings.warn(e.__str__())
                weights_initializer(module.weight)


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
    robot, G_in, G_out, G = get_robot_params(robot_name)

    # Parameters
    # activations = [Identity, torch.nn.SiLU, torch.nn.ReLU]
    # activations_names = ["Identity", "ReLU"]
    activations = [torch.nn.ReLU]
    activations_names = ["ReLU"]
    init_modes = ["fan_in", "fan_out", "arithmetic_mean", "harmonic_mean"]

    for activation, act_name in zip(activations, activations_names):
        df_activations, df_gradients, df_weights = None, None, None

        model_types = ['mlp', 'emlp']
        model_colors = plt.cm.get_cmap('inferno')([1.0, 0.0])
        alpha = 0.2
        for color, model_type in zip(model_colors, model_types):
            color[3] = alpha
            for i, init_mode in enumerate(init_modes):
                # Define output group for linear momentum
                if model_type == "emlp":
                    network = EMLP(rep_in=Vector(G_in), rep_out=Vector(G_out), group=G_in, num_layers=cfg.num_layers,
                                   ch=cfg.num_channels, with_bias=False, activation=activation, init_mode=init_mode,
                                   cache_dir=cache_dir).to(dtype=torch.float32)
                    network.save_cache_file()
                elif model_type == 'mlp':
                    if "mean" in init_mode: continue
                    network = MLP(d_in=G_in.d, d_out=G_out.d, num_layers=cfg.num_layers, ch=cfg.num_channels,
                                  with_bias=False, activation=activation, init_mode=init_mode).to(dtype=torch.float32)
                else:
                    raise NotImplementedError(model_type)

                dataset = COMMomentum(robot, cfg.dataset.num_samples, angular_momentum=cfg.dataset.angular_momentum,
                                      Gin=G_in, Gout=G_out, augment=False)
                data_loader = DataLoader(dataset, batch_size=cfg.batch_size)

                df_act = extract_activations(network, data_loader=data_loader)
                df_grad = extract_gradients(network, data_loader=data_loader)
                df_w = extract_weight_distribution(network)
                df_act["Model Type"] = model_type
                df_act["Initialization Mode"] = init_mode
                df_grad["Model Type"] = model_type
                df_grad["Initialization Mode"] = init_mode
                df_w["Model Type"] = model_type
                df_w["Initialization Mode"] = init_mode

                if df_activations is None:
                    df_activations, df_gradients, df_weights = df_act, df_grad, df_w
                else:
                    df_activations = pd.concat((df_activations, df_act), axis=0)
                    df_gradients = pd.concat((df_gradients, df_grad), axis=0)
                    df_weights = pd.concat((df_weights, df_w), axis=0)

        def plot_layers_distributions(df, value_kw, title=None, save=False):
            if "Param" in df.columns:
                df["hue"] = df["Model Type"] + "." + df["Param"]
            else:
                df["hue"] = df["Model Type"]
            g = sns.catplot(x="Layer", y=value_kw, hue="hue",
                            row="Initialization Mode", kind="violin", data=df,
                            sharey=False, sharex="col", aspect=(cfg.num_layers+2)/1.5, ci="sd",
                            scale='count', bw=.3, inner="box", scale_hue=True, pallete="Set3", leyend_out=False)
            if title:
                g.figure.suptitle(title)
                g.figure.subplots_adjust(top=0.92)
                if save:
                    g.figure.savefig(os.path.join(get_original_cwd(), "media", title), dpi=90)
                    print(f"Saving {title}")
            return g.figure

        save = cfg.num_layers > 7
        main_title = f"Layers={cfg.num_layers + 2}-Hidden_channels={cfg.num_channels}-Act={activation.__name__}"
        fig_grad = plot_layers_distributions(df=df_gradients, value_kw="Grad",
                                             title=f"Gradients Distributions " + main_title, save=save)
        fig_act = plot_layers_distributions(df=df_activations, value_kw="Activation",
                                            title=f"Activations Distributions " + main_title, save=save)
        fig_w = plot_layers_distributions(df=df_weights, value_kw="Param_Value",
                                          title=f"Params Distributions " + main_title, save=save)
        fig_grad.show()
        fig_act.show()
        fig_w.show()

        for model_type in model_types:
            main_title =f"{model_type}-Layers={cfg.num_layers+2}-Hidden_channels={cfg.num_channels}-Act={activation.__name__}"
            fig_grad = plot_layers_distributions(df=df_gradients[df_gradients["Model Type"] == model_type], value_kw="Grad",
                                                 title=f"Gradients Distributions " + main_title, save=save)
            fig_act = plot_layers_distributions(df=df_activations[df_activations["Model Type"] == model_type], value_kw="Activation",
                                                title=f"Activations Distributions " + main_title, save=save)
            fig_grad.show()
            fig_act.show()

        for fig in [fig_w, fig_act, fig_grad]:
            plt.close(fig)



if __name__ == "__main__":
    main()
