"""Cache and load the FFHQ arrays used by the repository's ALAE notebooks."""

import argparse
import fcntl
from pathlib import Path
import subprocess

import numpy as np


FFHQ_DIM = 512
TRAIN_SIZE = 60_000
TOTAL_SIZE = 70_000
DRIVE = 'https://drive.usercontent.google.com/download?confirm=t&id='
MODEL_DIR = Path('training_artifacts/ffhq')
MODEL_FILE = MODEL_DIR / 'model_157.pth'
FILES = {
    'age.npy': (DRIVE + '1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ', 560_128),
    'latents.npy': (DRIVE + '1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8', 143_360_128),
    'test_images.npy': (DRIVE + '1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i', 943_718_528),
    MODEL_FILE: ('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth', 651_591_168),
}


def download(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, (url, size) in FILES.items():
            path = directory / name
            if path.exists() and path.stat().st_size == size:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix('.part')
            print(f'Downloading {path}', flush=True)
            try:
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                '--retry', '3', '--connect-timeout', '30', '--output', str(partial),
                                url], check=True)
                if partial.stat().st_size != size:
                    raise ValueError(f'Incomplete download of {path}: expected {size:,} bytes')
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        marker = directory / MODEL_DIR / 'last_checkpoint'
        checkpoint = f'{directory / MODEL_FILE}\n'
        if not marker.exists() or marker.read_text() != checkpoint:
            partial = marker.with_suffix('.part')
            partial.write_text(checkpoint)
            partial.replace(marker)


def load_latents(directory):
    latents = np.load(directory / 'latents.npy', mmap_mode='r')
    ages = np.load(directory / 'age.npy', mmap_mode='r').reshape(-1)
    if latents.shape != (TOTAL_SIZE, FFHQ_DIM) or latents.dtype != np.float32 or ages.shape != (TOTAL_SIZE,):
        raise ValueError(f'Expected {TOTAL_SIZE:,} FFHQ codes and age labels in {directory}')
    return latents, ages


def check_cache(directory):
    directory = directory.resolve()
    missing = [str(name) for name, (_, size) in FILES.items()
               if not (directory / name).is_file() or (directory / name).stat().st_size != size]
    marker = directory / MODEL_DIR / 'last_checkpoint'
    if not marker.is_file() or marker.read_text().strip() != str(directory / MODEL_FILE):
        missing.append(str(marker.relative_to(directory)))
    if missing:
        raise SystemExit(
            f'Incomplete FFHQ cache at {directory}: {", ".join(missing)}.\n'
            f'From the repository root on a login node, run:\n'
            f'  python cc_experiments/ffhq.py "{directory}"'
        )
    print(f'FFHQ cache verified at {directory}')


def split_indices(ages, train):
    first, last = (0, TRAIN_SIZE) if train else (TRAIN_SIZE, TOTAL_SIZE)
    ages = ages[first:last]
    adult = np.flatnonzero(ages >= 18) + first
    child = np.flatnonzero((ages < 18) & (ages != -1)) + first
    if not (adult.size and child.size):
        raise ValueError('Expected labeled adults and children in the FFHQ split')
    return adult, child


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--check', action='store_true', help='Verify the cache offline')
    args = parser.parse_args()
    if args.check:
        check_cache(args.directory)
    else:
        download(args.directory)
