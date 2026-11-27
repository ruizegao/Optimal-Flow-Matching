"""Compare saved OFM transport maps on identical held-out inputs."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCRATCH = Path.home() / 'scratch/ofm_runs'
sys.path.insert(0, str(HERE.parent))
from cc_experiments.compare_results import METHODS
from cc_experiments.ffhq import check_cache
from cc_experiments.train import Batches, Config, new_model
from cc_experiments.visualize_ffhq import HELDOUT_SEED, decode_faces, heldout_faces, load_decoder


@dataclass
class Transport:
    method: str
    seed: int
    steps: int
    points: torch.Tensor

    @property
    def label(self):
        name = METHODS[self.method][0] if self.method in METHODS else self.method
        return f'{name}\nSeed {self.seed} · {self.steps:,} steps'


def checkpoints_under(directory):
    return {
        path.parent.relative_to(directory): path.resolve()
        for filename in ('model.pt', 'ckpt.pt')
        for path in directory.rglob(filename)
    }


def checkpoint_paths(inputs, dataset):
    if not inputs:
        roots = ((HERE / 'results', HERE / 'runs', SCRATCH / 'toy') if dataset == 'toy'
                 else (HERE / 'ffhq_results', SCRATCH / 'ffhq'))
        paths = {run: path for root in roots for run, path in checkpoints_under(root).items()}
        if not paths:
            raise ValueError(f'No {dataset} models found. Pass a run directory or checkpoint.')
        return sorted(paths.values())

    paths = set()
    for path in inputs:
        if path.is_file():
            paths.add(path.resolve())
        else:
            found = checkpoints_under(path)
            if not found:
                raise ValueError(f'No model.pt or ckpt.pt files under {path}')
            paths.update(found.values())
    return sorted(paths)


def transport_maps(paths, source, dataset, seed):
    ffhq = dataset == 'ffhq'
    transports = []
    for path in paths:
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = Config(**state['config'])
        if (config.ffhq_data is not None) != ffhq:
            raise ValueError(f'{path} does not belong to the {dataset} dataset')
        if seed is not None and config.seed != seed:
            continue
        model = new_model(config.width, config.batch_size, source.device, True, ffhq=ffhq)
        model.load_state_dict(state['model'])
        model.eval().requires_grad_(False)
        points = model.push_nograd(source).cpu()
        if not points.isfinite().all().item():
            raise ValueError(f'{path} returned nonfinite transport coordinates')
        transports.append(Transport(config.method, config.seed, state['next_iteration'], points))
    if not transports:
        raise ValueError(f'No {dataset} checkpoints match seed {seed}')
    order = {method: i for i, method in enumerate(METHODS)}
    return sorted(transports, key=lambda run: (order.get(run.method, len(order)), run.method,
                                               run.seed, run.steps))


def toy_figure(source, target, transports, full_range):
    source, target = source.cpu().numpy(), target.numpy()
    angle = np.arctan2(source[:, 1], source[:, 0])
    sectors = (np.floor((angle + np.pi / 8) / (np.pi / 4)).astype(int)) % 8
    colors = np.array(plt.get_cmap('tab10').colors)[sectors]
    starts = source[:32]
    panels = [('Held-out source', source, colors), ('Held-out target', target, '#575E69')]
    panels.extend((run.label, run.points.numpy(), colors) for run in transports)
    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.35 * rows + .6),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    extent = panels if full_range else panels[:2]
    limit = 1.1 * max(np.abs(points).max() for _, points, _ in extent)
    for i, (ax, (title, points, color)) in enumerate(zip(axes.flat, panels)):
        ax.set_title(title, fontsize=10)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal')
        if i >= 2:
            delta = points[:len(starts)] - starts
            ax.quiver(*starts.T, *delta.T, angles='xy', scale_units='xy', scale=1,
                      color='#283445', alpha=.38, width=.0045)
        ax.scatter(*points.T, c=color, s=5, linewidths=0, alpha=.62)
        ax.grid(alpha=.18)
        outside = (np.abs(points) > limit).any(axis=1).sum()
        if outside:
            ax.text(.97, .97, f'{outside} outside view', transform=ax.transAxes, ha='right',
                    va='top', fontsize=8, color='#575E69')
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle('Gaussian → Eight Gaussians')
    caption = 'Colors track source sectors. Arrows show the displacement of the same source points.'
    if not full_range:
        caption += '\nAxes follow the held-out source and target ranges.'
    fig.supxlabel(caption, fontsize=9)
    return fig


def ffhq_figure(originals, transports, data, device):
    decoder = load_decoder(data, device)
    rows, columns = len(transports) + 1, len(originals)
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns + 1.9, 1.85 * rows + .5),
                             squeeze=False, layout='constrained')
    for row, ax_row in enumerate(axes):
        if row == 0:
            label, faces = 'Held-out adults', originals
        else:
            run = transports[row - 1]
            label, faces = run.label, decode_faces(decoder, run.points.to(device))
        for ax, face in zip(ax_row, faces):
            ax.imshow(face)
            ax.set(xticks=[], yticks=[], frame_on=False)
        ax_row[0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=9)
    fig.suptitle('FFHQ adult → child')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('toy', 'ffhq'))
    parser.add_argument('inputs', nargs='*', type=Path, help='Saved models, checkpoints, or run directories')
    parser.add_argument('--seed', type=int, help='Show only runs with this training seed')
    parser.add_argument('--samples', type=int, help='Number of inputs (default: 1024 toy, 6 FFHQ)')
    parser.add_argument('--full-range', action='store_true', help='Include toy outliers in the axis limits')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--ffhq-data', type=Path, default=SCRATCH / 'ffhq_data')
    parser.add_argument('--output', type=Path, help='Output image file')
    args = parser.parse_intermixed_args()
    ffhq = args.dataset == 'ffhq'
    samples = args.samples if args.samples is not None else (6 if ffhq else 1024)
    if samples < 1:
        parser.error('--samples must be positive')
    if ffhq and args.full_range:
        parser.error('--full-range applies to toy plots')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu or run on a GPU node.')

    try:
        paths = checkpoint_paths(args.inputs, args.dataset)
        if ffhq:
            check_cache(args.ffhq_data)
            source, originals = heldout_faces(args.ffhq_data, samples)
        else:
            source, target = Batches(HELDOUT_SEED).endpoints(samples)
        transports = transport_maps(paths, source.to(args.device), args.dataset, args.seed)
        fig = (ffhq_figure(originals, transports, args.ffhq_data, args.device) if ffhq
               else toy_figure(source, target, transports, args.full_range))
        output = args.output or SCRATCH / f'{args.dataset}_transport_maps.png'
        if not output.suffix:
            output = output.with_suffix('.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=170)
        plt.close(fig)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Wrote {output} for {len(transports)} checkpoints')


if __name__ == '__main__':
    main()
