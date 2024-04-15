import copy
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

from utils.algebra_utils import cm2inch

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def merge_hps(d: dict):
    new_tr = copy.copy(d)
    d['Model Type'] = d['model'] + ('-aug' if d['dataset.augment'] == 'True' else '')
    new_tr.pop('model')
    new_tr.pop('dataset.augment')
    # new_tr.pop('hc')
    # TR
    new_tr.pop('train_ratio')
    str_hps = pretty_hps(new_tr)
    new_tr['Model Type'] = d['Model Type']
    new_tr['Hyper params'] = str_hps
    new_tr['train_ratio'] = d['train_ratio']
    return new_tr


def pretty_hps(d: dict):
    str_list = []
    d_sorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}
    for key, val in d_sorted.items():
        str_list.append(f'{key}={val}')

    translations = {"dataset.augment=True": "Aug",
                    "dataset.augment=False": "",
                    "finetuned=True": 'fine',
                    "finetuned=False": '',
                    'model.inv_dims_scale': 'inv',
                    'model.lr=': 'lr',
                    'G=C2': r'$\mathcal{G}=\mathcal{C}_2$',
                    'G=K4': r'$\mathcal{G}=\mathcal{K}_4$',
                    '0.0001': '1e-4',
                    '1e-05': '1e-5',
                    'train_ratio=': 'tr',
                    'hc=': 'hc',
                    "model=": ""}
    translations = dict((re.escape(k), v) for k, v in translations.items())
    # Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
    pattern = re.compile("|".join(translations.keys()))

    text = '-'.join(str_list) if len(str_list) > 1 else str_list[0]
    pretty_text = pattern.sub(lambda m: translations[re.escape(m.group(0))], text)
    pretty_text = pretty_text.strip('-')
    return pretty_text


def get_relevant_parameters(*dirs):
    all_dir = {}
    rel_dir = {}
    for d in dirs:
        for key, val in d.items():
            if key in all_dir:
                all_dir[key].add(val)
            else:
                all_dir[key] = {val}

    for key, val in all_dir.items():
        if len(val) > 1 or key == 'model':
            rel_dir[key] = val
    rel_dir.pop('seed')
    return rel_dir


def split_run_name(run_name):
    split = re.split('/| ', run_name)
    dirs = [a for a in split if '=' not in a and not 'version' in a and not 'experiments' in a and not 'metrics' in a]
    hp = [a for a in split if '=' in a]

    # Fix naming fuckup
    f = []
    for p in hp:
        parts = p.split('=')
        for part in parts:
            if '_' in part and 'model.' not in part and 'dataset.' not in part and 'volatile' not in part:
                v, k = part.split('_', maxsplit=1)
                f.extend([v, k])
            else:
                f.append(part)
    hp_dict = {}
    for k, v in zip(f[0::2], f[1::2]):
        hp_dict[k] = v

    if 'finetuned' not in hp_dict:
        hp_dict['finetuned'] = False

    return hp_dict, dirs


if __name__ == "__main__":
    print(sys.argv, len(sys.argv))

    # experiments_path = 'experiments/com_sample_eff_Atlas-C2'
    experiments_path = 'experiments/com_sample_eff_Solo-K4-C2'
    ignore_hps = [# 'train_ratio=0.5',
                  # 'train_ratio=0.6',
                  # 'train_ratio=0.7',
                  ['scale=0.0', 'augment=True'],
                  'hc=64', 'hc=128',
                  'hc=256',
                  # 'scale=0.0',
                  'scale=0.1',
                  'scale=0.25',
                  'scale=0.5',
                  'scale=1.0',
                  'scale=1.5',
                  'scale=2.0',
                  'scale=2.5',
                  'finetuned=True'
                  ]
    # ignore_hps = ['finetuned', 'scale=0.25', 'scale=0.5', 'scale=1.0', 'scale=1.5', 'scale=2.0', 'scale=2.5']
    # filter_hps = ['hc=512']
    # ignore_hps = []
    metrics_filter = ['err', 'cos', 'LH', 'RH', 'LF', 'RF',]
    filter_hps = [#"use_volatile=True",
                  #'model=ECNN',
                  # 'augment=True',
                  #'model=EMLP',
                  ]
    unique_hps = {'finetuned': {True, False},
                  'dataset.augment': {True, False}}

    print(f"ignoring runs with {ignore_hps}")
    print(f"Filtering runs by {filter_hps}")

    experiments_path = pathlib.Path(experiments_path)
    assert experiments_path.exists()

    metrics_paths = []
    all_run_paths = list(experiments_path.rglob("*/test_metrics.csv"))

    # Filter run values
    exp_metrics = None
    hps, dirs = [], []
    for path in all_run_paths:
        assert isinstance(path, pathlib.Path)
        should_ignore = False
        for ignore_hp in ignore_hps:
            is_hp_preset = lambda x: x in str(path)
            if isinstance(ignore_hp, str):
                should_ignore = is_hp_preset(ignore_hp)
            elif isinstance(ignore_hp, list):
                present = [is_hp_preset(hp) for hp in ignore_hp]
                should_ignore = np.all(present)
            else:
                raise NotImplementedError()
            if should_ignore: break
        a = should_ignore
        for filter_hp in filter_hps:
            if filter_hp not in str(path):
                should_ignore = True

        if should_ignore:
            continue
        metrics_paths.append(path)
        hp, dir = split_run_name(str(path))
        hps.append(hp)
        dirs.append(dir)


    runs = [p.parent.parent.parent for p in metrics_paths]
    u_runs, counts = np.unique(runs, return_counts=True)
    print(f"{np.sum(counts < 10)} runs are not complete")
    for run, n_seeds in zip(u_runs, counts):
        if n_seeds < 8:
            print(f"Warning. seeds={n_seeds} : {run}")

    # Errase irrelevant hyperparameters
    relevant_parameters = get_relevant_parameters(*hps)
    relevant_parameters.update(unique_hps)

    data = []
    for i, path in enumerate(metrics_paths):
        run_metrics = pd.read_csv(str(path), index_col=0)
        if exp_metrics is None:
            exp_metrics = list(run_metrics.columns)
        run_hps = hps[i]
        rel_hps = {k: v for k, v in run_hps.items() if k in relevant_parameters}
        merged_hps = merge_hps(rel_hps)
        for k, v in merged_hps.items():
            run_metrics[k] = v
        data.append(run_metrics)


    prefix = ("filter_" + str(filter_hps) if len(filter_hps) > 0 else '') + \
             ("ignore_" + str(ignore_hps) if len(ignore_hps) > 0 else '')
    out_path = experiments_path.joinpath('_'.join(['results', prefix]))
    out_path.mkdir(exist_ok=True)
    print(f"saving plots to {out_path}")

    # Agglomerate data and save single data frame.
    df = pd.concat(data, ignore_index=True)
    # Format all data as numeric.
    for c in exp_metrics + ['train_ratio', 'hc']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])

    # Squash all hyperparameters into single column
    plot_df = df[exp_metrics]
    # H, W = 12, 15
    H, W = 9, 10
    markers = [',', "o", '.']
    model_hps = list(df['Hyper params'].unique())
    model_hps.sort(key=natural_keys)
    model_types = sorted(df['Model Type'].unique())
    if 'ECNN' in model_types:
        model_types.reverse()

    for metric_name in exp_metrics:
        ignore = False
        for metric_filter in metrics_filter:
            if metric_filter in metric_name:
                ignore = True
        if ignore: continue

        if 'jaccard' in metric_name: continue
        fig, ax = plt.subplots(figsize=(cm2inch(W), cm2inch(H)), dpi=210)
        sns.lineplot(data=df, x='train_ratio', y=metric_name,
                     hue='Hyper params', hue_order=model_hps,
                     style='Model Type', style_order=model_types,
                     dashes=True, #markers=markers[:len(model_types)],
                     ax=ax, ci=100,
                     # palette=sns.color_palette("mako_r", len(model_hps)),
                     palette=sns.color_palette("dark:salmon_r", len(model_hps)),

                     )
        ax.grid(visible=True, alpha=0.2)
        ax.set(yscale='log')
        # ax.set(xscale='log')
        pretty_metric_name = metric_name.replace("_", " ")
        title = f'{experiments_path.stem}'
        fig_title = f'{experiments_path.stem}'.replace('sample_eff_', '').replace('com_', '')\
            .replace('contact_', '') \
            .replace('-K4-C2', r' - $\mathcal{G}=\mathcal{K}_4$ vs. $\mathcal{G}=\mathcal{C}_2$ ') \
            .replace('-K4', r' - $\mathcal{G}=\mathcal{K}_4$') \
            .replace('-C2', r' - $\mathcal{G}=\mathcal{C}_2$')
        ax.set_title(fig_title)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        # ax.legend(fancybox=True, framealpha=0.5)
        plt.legend(title=None, fontsize=7, fancybox=True, framealpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path / f'{title}.png')
        plt.show()

