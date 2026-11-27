"""Train the OFM solver comparisons on Eight Gaussians or FFHQ latents."""

import argparse
import csv
from collections import deque
from dataclasses import asdict, dataclass, replace
import fcntl
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cc_experiments.ffhq import FFHQ_DIM, load_latents, split_indices
from ofmsrc.icnn import DenseICNN, ICNN2CPF
from ofmsrc.ihv import BruteforceIHVParams, ofm_ihv_bruteforce
from ofmsrc.inverse import NesterovInvParams, TorchlbfgsInvParams, ofm_inverse
from ofmsrc.model_tools import id_pretrain_model
from ofmsrc.tools import EMA, freeze_context

# Batch and optimizer defaults follow the notebook.
METHODS = {
    'plain': {},
    'wolfe': {'wolfe': True},
    'retry': {'retry': True},
    'mlp': {'mlp': True},
    'mlp-retry': {'mlp': True, 'retry': True},
    'restricted': {'restricted': True},
    'restricted-wolfe': {'restricted': True, 'wolfe': True},
    'cap1000': {'cap': 1000},
    'nesterov': {'nesterov': True},
}
EPS = 1e-5
CENTERS = torch.tensor([
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [2**-.5, 2**-.5], [2**-.5, -2**-.5],
    [-2**-.5, 2**-.5], [-2**-.5, -2**-.5],
])


@dataclass
class Config:
    method: str
    seed: int = 42
    batch_size: int = 1024
    width: int = 1024
    hessian_chunk: int = 128
    pretrain_steps: int = 200
    tolerance: float = .01
    retry_displacement: float = 25.
    t_min: float = .01
    t_max: float = .85
    trace_inverse: bool = False
    ffhq_data: Path | None = None
    nesterov_velocity_tol: float = .1
    nesterov_max_iter: int = 300
    nesterov_psi_l: float | None = None


class Batches:
    def __init__(self, seed):
        self.torch = torch.Generator().manual_seed(seed)
        self.python = random.Random(seed)

    def state_dict(self):
        return self.torch.get_state(), self.python.getstate()

    def load_state_dict(self, state):
        self.torch.set_state(state[0].cpu())
        self.python.setstate(state[1])

    def endpoints(self, size):
        x0 = torch.randn(size, 2, generator=self.torch)
        x1 = .4 * torch.randn(size, 2, generator=self.torch)
        x1 += 3 * CENTERS[self.python.choices(range(len(CENTERS)), k=size)]
        return x0, x1

    def sample(self, size, device, interval=None):
        x0, x1 = self.endpoints(size)
        t = torch.rand(size, generator=self.torch)
        t = t + 1e-8 if interval is None else interval[0] + (interval[1] - interval[0]) * t
        x0, x1, t = (x.to(device) for x in (x0, x1, t))
        xt = x0 * (1 - t[:, None]) + x1 * t[:, None]
        return dict(X0=x0, X1=x1, Xt=xt, t=t)


class FFHQBatches(Batches):
    def __init__(self, seed, directory, train=True):
        super().__init__(seed)
        self.latents, ages = load_latents(directory)
        self.indices = split_indices(ages, train)

    def endpoints(self, size):
        samples = []
        for indices in self.indices:
            rows = torch.randint(len(indices), (size,), generator=self.torch).numpy()
            samples.append(torch.from_numpy(self.latents[indices[rows]]))
        return samples


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_atomic(payload, path):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def new_model(width, batch_size, device, initialized, ffhq=False):
    if ffhq:
        return DenseICNN(FFHQ_DIM, [width, width], batch_size=batch_size).to(device)
    return ICNN2CPF(dim=2, dimh=width, num_hidden_layers=2, batch_size=batch_size,
                    actnorm_initialized=initialized).to(device)


class Trainer:
    def __init__(self, config, device, state=None):
        self.config = config
        self.device = torch.device(device)
        options = METHODS[config.method]
        self.interval = (config.t_min, config.t_max) if options.get('restricted') else None
        ffhq = config.ffhq_data is not None
        dim = FFHQ_DIM if ffhq else 2
        self.data = FFHQBatches(config.seed, config.ffhq_data) if ffhq else Batches(config.seed)
        torch.manual_seed(config.seed)
        self.model = new_model(config.width, config.batch_size, self.device, state is not None, ffhq)
        if state is None and (config.pretrain_steps or not ffhq):
            scale = torch.full((dim,), 2. if ffhq else 4., device=self.device)
            sampler = torch.distributions.Normal(torch.zeros_like(scale), scale)
            if config.pretrain_steps:
                id_pretrain_model(self.model, sampler, n_max_iterations=config.pretrain_steps,
                                  batch_size=config.batch_size, lr=.01, verbose=False)
                self.model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    self.model(sampler.sample((config.batch_size,)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001 if ffhq else .01)
        self.ema = EMA(0, betas=[.999, .99])
        if options.get('nesterov'):
            psi_m = self.model.strong_convexity if ffhq else 0.
            self.inverse_params = NesterovInvParams(
                psi_L=psi_m if config.nesterov_psi_l is None else config.nesterov_psi_l,
                psi_m=psi_m, velocity_tol=config.nesterov_velocity_tol,
                max_iter=config.nesterov_max_iter, record_steps=config.trace_inverse,
                report_final_norms=True,
            )
        else:
            self.inverse_params = TorchlbfgsInvParams(
                lbfgs_params=dict(tolerance_grad=config.tolerance,
                                  line_search_fn='strong_wolfe' if options.get('wolfe') else None),
                max_iter=options.get('cap', 10), record_steps=config.trace_inverse,
                report_final_norms=True, report_solution=True,
                wolfe_retry_displacement=config.retry_displacement if options.get('retry') else None,
            )
        self.ihv_params = BruteforceIHVParams()
        self.amortizer = None
        if options.get('mlp'):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.seed + 1)
                self.amortizer = torch.nn.Sequential(
                    torch.nn.Linear(dim + 1, 64), torch.nn.SiLU(),
                    torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, dim),
                )
                torch.nn.init.zeros_(self.amortizer[-1].weight)
                torch.nn.init.zeros_(self.amortizer[-1].bias)
            self.amortizer.to(self.device)
            self.amort_optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-3)
        self.next_iteration = 0
        self.compute_seconds = 0.
        self.recent = deque(maxlen=100)
        self.retries = 0
        if state is not None:
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.ema._step = state['next_iteration']
            if state['ema']:
                self.ema.model_copies = [new_model(config.width, config.batch_size, self.device, True, ffhq)
                                         for _ in self.ema.betas]
                for model, weights in zip(self.ema.model_copies, state['ema']):
                    model.load_state_dict(weights)
            if self.amortizer is not None:
                self.amortizer.load_state_dict(state['amortizer'])
                self.amort_optimizer.load_state_dict(state['amort_optimizer'])
            self.next_iteration = state['next_iteration']
            self.compute_seconds = state['compute_seconds']
            self.recent.extend(state['recent'])
            self.retries = state['retries']
            self.data.load_state_dict(state['data_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if self.device.type == 'cuda' and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state(state['cuda_rng'].cpu(), self.device)

    def state_dict(self):
        state = dict(config=asdict(self.config), next_iteration=self.next_iteration,
                     model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     ema=[m.state_dict() for m in getattr(self.ema, 'model_copies', [])],
                     recent=list(self.recent), retries=self.retries, compute_seconds=self.compute_seconds,
                     data_rng=self.data.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None)
        if self.amortizer is not None:
            state.update(amortizer=self.amortizer.state_dict(),
                         amort_optimizer=self.amort_optimizer.state_dict())
        return state

    def batch(self):
        return self.data.sample(self.config.batch_size, self.device, self.interval)

    def estimate_psi_l(self, xt, psi_m):
        # Probe the current potential at the origin and at inputs from this batch.
        # The factor of two cushions the sampled estimate against missed curvature.
        count = 3 if self.config.ffhq_data is not None else 15
        points = torch.cat((xt[:count], torch.zeros_like(xt[:1])))
        with freeze_context(self.model):
            hessians = torch.cat([
                self.model.hessian_nograd(p).squeeze(1)
                for p in points.split(self.config.hessian_chunk)
            ])
        if not hessians.isfinite().all().item():
            raise FloatingPointError('Non-finite Hessian in Nesterov smoothness estimate')
        return max(psi_m, 2. * torch.linalg.eigvalsh(hessians)[..., -1].max().item(), 1e-6)

    def prepare(self, batch, inverse=None):
        """Solve the inverse and build the OFM loss before mutating optimizer state."""
        t, xt = batch['t'], batch['Xt']
        z_init = None
        amort_time = 0.
        params = self.inverse_params
        if self.amortizer is not None:
            started = time.perf_counter()
            z_init = xt + t[:, None] * self.amortizer(torch.cat((xt, t[:, None]), dim=1))
            params = replace(params, initial_guess=z_init.detach())
            sync(self.device)
            amort_time = time.perf_counter() - started

        started = time.perf_counter()
        stats = {}
        if isinstance(params, NesterovInvParams) and inverse is None:
            stats['time_bound'] = 0.
            if self.config.nesterov_psi_l is None:
                params = replace(params, psi_L=self.estimate_psi_l(xt, params.psi_m))
                stats['time_bound'] = time.perf_counter() - started
        z = ofm_inverse(self.model, xt, t, params, stats) if inverse is None else inverse
        sync(self.device)
        inverse_time = time.perf_counter() - started
        v = 2 * (batch['X0'] - z) / (t[:, None] + EPS)
        sample_loss = v.square().sum(1)
        true_loss = sample_loss.mean().item()

        started = time.perf_counter()
        if math.isfinite(true_loss):
            # Chunk the expensive vmapped Hessian independently of the inverse solve.
            size = self.config.hessian_chunk
            with freeze_context(self.model):
                lhs = torch.cat([
                    ofm_ihv_bruteforce(self.model, zz.detach().clone(), vv, tt, self.ihv_params)
                    for zz, vv, tt in zip(z.split(size), v.split(size), t.split(size))
                ])
            sync(self.device)
            ihv_time = time.perf_counter() - started
            rhs = self.model.push(z.detach())
            proxy = (lhs * rhs).sum(1).mean()
        else:
            ihv_time = 0.
            proxy = v.new_tensor(float('nan'))
        return dict(z=z.detach(), z_init=z_init, solver=stats, true_loss=true_loss,
                    proxy=proxy, loss=proxy.item(), sample_loss=sample_loss.detach(),
                    t_inverse=inverse_time, t_ihv=ihv_time, t_amort=amort_time)

    def update(self, batch, step, teacher=None, track_update=False):
        if not (math.isfinite(step['true_loss']) and math.isfinite(step['loss'])):
            raise FloatingPointError('Non-finite OFM loss before Adam update')
        before = [p.detach().clone() for p in self.model.parameters()] if track_update else None
        self.optimizer.zero_grad(set_to_none=True)
        step['proxy'].backward()
        grad_norm = float('nan')
        update_norm = float('nan')
        if track_update:
            grad_norm = math.sqrt(sum(p.grad.detach().square().sum().item()
                                      for p in self.model.parameters() if p.grad is not None))
        self.optimizer.step()
        self.model.convexify()
        if track_update:
            update_norm = math.sqrt(sum((p.detach() - previous).square().sum().item()
                                        for p, previous in zip(self.model.parameters(), before)))
        self.ema(self.model)

        amort_loss, target_fraction = float('nan'), 0.
        if self.amortizer is not None:
            started = time.perf_counter()
            target, residuals = ((step['z'], step['solver']['norms']) if teacher is None else teacher)
            valid = residuals <= self.config.tolerance
            solved = batch['t'] >= self.inverse_params.t_min
            count = valid.sum().item()
            if count:
                loss = (step['z_init'][solved][valid] - target[solved][valid]).square().sum(1).mean()
                self.amort_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.amort_optimizer.step()
                amort_loss = loss.item()
            target_fraction = count / len(solved)
            sync(self.device)
            step['t_amort'] += time.perf_counter() - started
        self.next_iteration += 1
        return dict(amort_loss=amort_loss, amort_target_fraction=target_fraction,
                    grad_norm=grad_norm, parameter_update_norm=update_norm)

    def evaluate_map(self, size=1024):
        data = (FFHQBatches(8191, self.config.ffhq_data, train=False)
                if self.config.ffhq_data is not None else Batches(8191))
        heldout = data.sample(size, self.device)
        prediction = self.model.push_nograd(heldout['X0'])
        return sliced_w2(prediction, heldout['X1'])


def sliced_w2(prediction, target):
    dim = target.shape[1]
    if dim == 2:
        angle = torch.arange(128, device=target.device, dtype=target.dtype) * (math.pi / 128)
        directions = torch.stack((angle.cos(), angle.sin()))
    else:
        generator = torch.Generator().manual_seed(8191)
        directions = torch.randn(dim, 128, generator=generator, dtype=target.dtype).to(target.device)
        directions = torch.nn.functional.normalize(directions, dim=0)
    return ((prediction @ directions).sort(dim=0).values
            - (target @ directions).sort(dim=0).values).square().mean().sqrt().item()


def step_metrics(batch, step, trainer):
    s = step['solver']
    metrics = dict(
        loss=step['loss'], true_loss=step['true_loss'], t_inverse=step['t_inverse'],
        t_ihv=step['t_ihv'], t_amort=step['t_amort'], t_diag=s.get('time_diag', 0.),
        inverse_steps=s.get('steps', 0), inner_iterations=s.get('inner_iterations', 0),
        inverse_func_evals=s.get('func_evals', 0), inverse_max_residual=s.get('norms_max', float('nan')),
        inverse_max_displacement=(step['z'] - batch['Xt']).norm(dim=1).max().item(),
        t_min=batch['t'].min().item(), t_max=batch['t'].max().item(),
        wolfe_retried=int(s.get('wolfe_retried', False)),
        plain_func_evals=s.get('plain_func_evals', 0), wolfe_func_evals=s.get('wolfe_func_evals', 0),
        plain_steps=s.get('plain_steps', 0), wolfe_steps=s.get('wolfe_steps', 0),
        plain_max_displacement=s.get('plain_max_displacement', float('nan')),
        cap_hit=int(max(s.get('steps', 0), s.get('plain_steps', 0)) >= trainer.inverse_params.max_iter),
    )
    if isinstance(trainer.inverse_params, NesterovInvParams):
        metrics.update(
            cap_hit=int(s.get('cap_hit', False)),
            inverse_certified_fraction=s.get('converged', math.nan),
            inverse_max_velocity_bound=s.get('velocity_bound_max', math.nan),
            inverse_gradient_samples=s.get('gradient_samples', math.nan),
            inverse_psi_l=s.get('psi_L', math.nan),
            t_inverse_bound=s.get('time_bound', math.nan),
        )
    return metrics


def csv_resume(path, next_iteration):
    if not path.exists():
        if next_iteration:
            raise FileNotFoundError(f'Missing profile for checkpoint at {next_iteration}')
        return deque()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get('iteration') and row['iteration'].isdigit()]
        header = reader.fieldnames
    valid = [row for row in rows if int(row['iteration']) < next_iteration]
    if [int(row['iteration']) for row in valid] != list(range(next_iteration)):
        raise ValueError(f'Profile does not cover checkpoint history in {path}')
    if header is None:
        return deque()
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(valid)
    os.replace(tmp, path)
    return deque(int(row['wolfe_retried']) for row in valid)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--report-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--spike-factor', type=float, default=4.)
    p.add_argument('--output', type=Path)
    p.add_argument('--ffhq-data', type=Path, help='Directory containing the Figure 7 FFHQ arrays')
    p.add_argument('--batch-size', type=int, help='Default: 128 for FFHQ, 1024 for Eight Gaussians')
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--hessian-chunk', type=int,
                   help='Rows per Hessian calculation. The L-BFGS batch stays whole.')
    p.add_argument('--pretrain-steps', type=int, help='Default: 0 for FFHQ, 200 for Eight Gaussians')
    p.add_argument('--tolerance', type=float, default=.01)
    p.add_argument('--retry-displacement', type=float, default=25.)
    p.add_argument('--t-min', type=float, default=.01, help='Lower time bound for restricted methods')
    p.add_argument('--t-max', type=float, default=.85, help='Upper time bound for restricted methods')
    p.add_argument('--trace-inverse', action='store_true', help='Record per-step residuals at additional cost')
    p.add_argument('--capture-iteration', type=int, help='Also capture this zero-based iteration')
    p.add_argument('--nesterov-velocity-tol', type=float, default=Config.nesterov_velocity_tol,
                   help='Euclidean velocity tolerance for Nesterov (default: %(default)s)')
    p.add_argument('--nesterov-max-iter', type=int, default=Config.nesterov_max_iter,
                   help='Maximum Nesterov updates per sample (default: %(default)s)')
    p.add_argument('--nesterov-psi-l', type=float,
                   help='Fixed potential smoothness bound. Defaults to a per-batch estimate.')
    args = p.parse_args()
    ffhq = args.ffhq_data is not None
    if ffhq:
        args.ffhq_data = args.ffhq_data.resolve()
    if args.output is None:
        args.output = (Path.home() / 'scratch/ofm_runs' / ('ffhq' if ffhq else 'toy')
                       if ffhq or args.method == 'nesterov' else Path(__file__).parent / 'runs')
    if args.batch_size is None:
        args.batch_size = 128 if ffhq else 1024
    if args.hessian_chunk is None:
        args.hessian_chunk = 8 if ffhq else 128
    if args.pretrain_steps is None:
        args.pretrain_steps = 0 if ffhq else 200
    if (args.batch_size < 2 or min(args.iterations, args.width, args.hessian_chunk,
                                  args.checkpoint_every, args.report_every, args.eval_every) < 1
            or args.pretrain_steps < 0 or not 0 <= args.t_min < args.t_max <= 1
            or not 0 < args.tolerance < math.inf or not 0 < args.retry_displacement < math.inf
            or not 1 <= args.spike_factor < math.inf):
        p.error('Invalid batch size, time bounds, tolerance, or run interval')
    if not 0. < args.nesterov_velocity_tol < math.inf or args.nesterov_max_iter < 0:
        p.error('Use a positive finite Nesterov velocity tolerance and a nonnegative step cap')
    if args.nesterov_psi_l is not None and not 0. <= args.nesterov_psi_l < math.inf:
        p.error('Use a nonnegative finite Nesterov smoothness bound')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable. For a local smoke test use --device cpu.')
    return args


def train(args, directory):
    config = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    checkpoint = directory / 'ckpt.pt'
    profile = directory / 'profile.csv'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if state is not None and Config(**state['config']) != config:
        raise ValueError('Training settings differ from the checkpoint. Choose a new --output directory.')
    events = directory / 'spikes'
    decisions = deque(csv_resume(profile, state['next_iteration'] if state is not None else 0),
                      maxlen=args.report_every)
    trainer = Trainer(config, args.device, state)
    device = trainer.device
    if device.type == 'cuda':
        gpu = torch.cuda.get_device_properties(device)
        print(f'{gpu.name}, {gpu.total_memory / 2**30:.2f} GiB visible')
    print(f'{config.method}, seed {config.seed}, iterations {trainer.next_iteration}–{args.iterations - 1}, '
          f'batch {config.batch_size}, Hessian chunk {config.hessian_chunk}')

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)

    with profile.open('a', newline='') as f:
        writer = None
        for i in range(trainer.next_iteration, args.iterations):
            if stop:
                break
            started = time.perf_counter()
            batch = trainer.batch()
            try:
                step = trainer.prepare(batch)
            except (FloatingPointError, torch.linalg.LinAlgError) as error:
                events.mkdir(exist_ok=True)
                save_atomic(dict(state=trainer.state_dict(), batch=batch, error=str(error)),
                            events / f'iter_{i:06d}.pt')
                raise
            metrics = step_metrics(batch, step, trainer)
            baseline = statistics.median(trainer.recent) if len(trainer.recent) >= 20 else float('nan')
            ratio = step['true_loss'] / baseline if baseline > 0 else float('nan')
            capture = (ratio >= args.spike_factor or i == args.capture_iteration
                       or not (math.isfinite(step['true_loss']) and math.isfinite(step['loss'])))
            capture_overhead = 0.
            if capture:
                capture_started = time.perf_counter()
                events.mkdir(exist_ok=True)
                event = events / f'iter_{i:06d}.pt'
                save_atomic(dict(state=trainer.state_dict(), batch=batch, z=step['z'], solver=step['solver'],
                                 z_init=step['z_init'].detach() if step['z_init'] is not None else None,
                                 metrics=metrics, ratio=ratio, baseline=baseline), event)
                capture_overhead = time.perf_counter() - capture_started
                print(f'Captured iteration {i}, true loss {step["true_loss"]:.5g}, ratio {ratio:.3g}, {event}')
            updates = trainer.update(batch, step, track_update=capture)
            sync(device)
            elapsed = time.perf_counter() - started - capture_overhead
            trainer.compute_seconds += elapsed
            trainer.recent.append(step['true_loss'])
            decisions.append(metrics['wolfe_retried'])
            trainer.retries += metrics['wolfe_retried']
            metrics['t_amort'] = step['t_amort']
            window = list(decisions)[-args.report_every:] if trainer.next_iteration % args.report_every == 0 else []
            report = bool(window) or stop or trainer.next_iteration == args.iterations
            if report and not window:
                partial = trainer.next_iteration % args.report_every
                window = list(decisions)[-(partial or args.report_every):]
            eval_due = (trainer.next_iteration % args.eval_every == 0
                        or trainer.next_iteration == args.iterations or stop)
            evaluation = trainer.evaluate_map() if eval_due else float('nan')
            row = dict(
                iteration=i, **metrics, **updates, t_total=elapsed, train_seconds=trainer.compute_seconds,
                spike_ratio=ratio, spike_captured=int(capture), heldout_sliced_w2=evaluation,
                switching_ratio=sum(window) / len(window) if window else float('nan'),
                cumulative_switching_ratio=trainer.retries / trainer.next_iteration,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0.,
            )
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row)
                if f.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            save_due = trainer.next_iteration % args.checkpoint_every == 0
            if report or save_due:
                f.flush()
            if report:
                if config.method == 'nesterov':
                    solver_report = (f'Nesterov certified {metrics["inverse_certified_fraction"]:.2%}, '
                                     f'mean steps {metrics["inverse_steps"]:.2f}, '
                                     f'psi_L {metrics["inverse_psi_l"]:.4g}')
                else:
                    solver_report = (f'Strong Wolfe retries {sum(window)}/{len(window)}, '
                                     f'cumulative {row["cumulative_switching_ratio"]:.2%}')
                print(f'Iteration {i}, true loss {step["true_loss"]:.4g}, '
                      f'{solver_report}, train {trainer.compute_seconds:.1f}s, '
                      f'peak CUDA reserved {row["cuda_peak_reserved_mib"]:.0f} MiB')
            if save_due:
                save_atomic(trainer.state_dict(), checkpoint)
        f.flush()
        save_atomic(trainer.state_dict(), checkpoint)
    print(f'Saved {checkpoint} at iteration {trainer.next_iteration}')


def main():
    args = arguments()
    directory = args.output / args.method / f'seed_{args.seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'A job already owns {directory}')
        train(args, directory)


if __name__ == '__main__':
    main()
