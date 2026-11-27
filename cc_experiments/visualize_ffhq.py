"""Render the same held-out FFHQ adults and their translations for each solver."""

import argparse
from importlib.util import find_spec
from pathlib import Path
import site
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cc_experiments.ffhq import MODEL_DIR, TRAIN_SIZE, load_latents, split_indices
from cc_experiments.train import Config, new_model


HELDOUT_SEED = 8191


def heldout_faces(data, count=8):
    latents, ages = load_latents(data)
    adult, _ = split_indices(ages, train=False)
    images = np.load(data / 'test_images.npy', mmap_mode='r')
    available = adult[adult - TRAIN_SIZE < len(images)]
    if not 1 <= count <= len(available):
        raise ValueError(f'Choose between 1 and {len(available)} held-out adult faces')
    indices = np.random.default_rng(HELDOUT_SEED).choice(available, count, replace=False)
    return torch.from_numpy(latents[indices]), images[indices - TRAIN_SIZE]


def load_decoder(data, device):
    alae_dir = Path(__file__).resolve().parents[1] / 'ALAE'
    sys.path.insert(0, str(alae_dir))
    # The cluster module disables user-site imports, where yacs may be installed.
    if find_spec('yacs') is None and not site.ENABLE_USER_SITE:
        sys.path.append(site.getusersitepackages())
    from alae_ffhq_inference import load_model
    return load_model(str(alae_dir / 'configs/ffhq.yaml'), str(data / MODEL_DIR)).to(device)


@torch.no_grad()
def decode_faces(decoder, mapped):
    from alae_ffhq_inference import decode
    torch.manual_seed(HELDOUT_SEED)
    images = decode(decoder, mapped)
    return images.mul_(.5).add_(.5).clamp_(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def visualize(checkpoint, device, data=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = Config(**state['config'])
    if config.ffhq_data is None:
        raise ValueError(f'{checkpoint} is not an FFHQ run')
    data = data if data is not None else Path(config.ffhq_data)
    source, originals = heldout_faces(data)
    model = new_model(config.width, config.batch_size, device, True, ffhq=True)
    model.load_state_dict(state['model'])
    model.eval().requires_grad_(False)
    mapped = model.push_nograd(source.to(device))
    translated = decode_faces(load_decoder(data, device), mapped)

    fig, axes = plt.subplots(2, len(originals), figsize=(14, 4), layout='constrained')
    for column, (original, child) in enumerate(zip(originals, translated)):
        for row, image in enumerate((original, child)):
            ax = axes[row, column]
            ax.imshow(image)
            ax.set(xticks=[], yticks=[], frame_on=False)
    axes[0, 0].set_ylabel('Held-out adult')
    axes[1, 0].set_ylabel('OFM translation')
    fig.suptitle(f'FFHQ adult → child · {config.method} · seed {config.seed} · '
                 f'{state["next_iteration"]:,} training steps')
    output = checkpoint.with_name('translations.png')
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f'Wrote {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--ffhq-data', type=Path, help='Override the FFHQ asset directory stored in the checkpoint')
    args = parser.parse_args()
    visualize(args.checkpoint, args.device, args.ffhq_data)
