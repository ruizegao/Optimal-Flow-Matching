"""Inspect a saved spike, replay its Adam update, or compare completed run profiles."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import torch

from train import (Batches, Config, EPS, Trainer, new_model, sliced_w2, step_metrics, sync)


def refine(model, batch, chunk):
    """Float64 damped Newton, with a strong-convexity bound for t < 1."""
    xt, x0, t = (batch[k].double().cpu() for k in ('Xt', 'X0', 't'))
    model.eval().requires_grad_(False)
    z = xt.clone()
    identity = torch.eye(z.shape[1], dtype=torch.float64)

    def gradient(pos, target, ti):
        return (1 - ti[:, None]) * pos + ti[:, None] * model.push_nograd(pos.detach()) - target

    def matrices(pos, ti):
        h = torch.cat([model.hessian_nograd(p.detach()).squeeze(1) for p in pos.split(chunk)])
        return ti[:, None, None] * h + (1 - ti[:, None, None]) * identity

    scale = 2 / ((t + EPS) * (1 - t))
    g = gradient(z, xt, t)
    for _ in range(60):
        active = torch.where(g.norm(dim=1) * scale > 1e-7)[0]
        if not len(active):
            break
        pos, ta = z[active], t[active]
        direction, info = torch.linalg.solve_ex(matrices(pos, ta), g[active, :, None])
        direction = direction.squeeze(-1)
        pending = torch.where((info == 0) & direction.isfinite().all(1))[0]
        accepted = 0
        for j in range(24):
            if not len(pending):
                break
            idx = active[pending]
            trial = pos[pending] - 0.5**j * direction[pending]
            new_g = gradient(trial, xt[idx], t[idx])
            good = new_g.norm(dim=1) < g[idx].norm(dim=1)
            z[idx[good]] = trial[good]
            g[idx[good]] = new_g[good]
            accepted += int(good.sum())
            pending = pending[~good]
        if not accepted:
            break

    residual = g.norm(dim=1)
    bound = residual * scale
    bound[~bound.isfinite()] = torch.inf
    v = 2 * (x0 - z) / (t[:, None] + EPS)
    return z, v, residual, bound, matrices, gradient


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def refine_report(model, batch, observed, chunk):
    z, v, residual, bound, matrices, gradient = refine(model, batch, chunk)
    norm = v.norm(dim=1)
    loss = norm.square()
    summary = dict(
        refined_true_loss=loss.mean().item(), max_refined_residual=residual.max().item(),
        max_loss_vector_error_bound=bound.max().item(),
        loss_lower_bound=(norm - bound).clamp_min(0).square().mean().item(),
        loss_upper_bound=(norm + bound).square().mean().item(),
    )
    if observed is None:
        return z, summary, []
    t = batch['t'].double()
    observed = observed.double()
    raw = (2 * (batch['X0'].double() - observed) / (t[:, None] + EPS)).square().sum(1)
    summary.update(observed_true_loss=raw.mean().item(),
                   observed_minus_refined=raw.mean().item() - loss.mean().item())
    finite = observed.isfinite().all(1)
    eig = torch.full((len(t), 2), float('nan'), dtype=torch.float64)
    initial_residual = torch.full_like(t, float('nan'))
    if finite.any():
        eig[finite] = torch.linalg.eigvalsh(matrices(observed[finite], t[finite]))
        initial_residual[finite] = gradient(observed[finite], batch['Xt'][finite].double(), t[finite]).norm(dim=1)
    refined_eig = torch.linalg.eigvalsh(matrices(z, t))
    rows = [
        dict(sample=i, t=t[i].item(), observed_loss=raw[i].item(), refined_loss=loss[i].item(),
             loss_difference=(raw[i] - loss[i]).item(),
             observed_z0_x=observed[i, 0].item(), observed_z0_y=observed[i, 1].item(),
             refined_z0_x=z[i, 0].item(), refined_z0_y=z[i, 1].item(),
             observed_residual=initial_residual[i].item(), refined_residual=residual[i].item(),
             observed_min_eigenvalue=eig[i, 0].item(), observed_max_eigenvalue=eig[i, 1].item(),
             refined_min_eigenvalue=refined_eig[i, 0].item(), refined_max_eigenvalue=refined_eig[i, 1].item())
        for i in raw.argsort(descending=True).tolist()
    ]
    return z, summary, rows


def follow_branch(event, z, label, device, steps, path):
    state = event['state']
    trainer = Trainer(Config(**state['config']), device, state)
    batch = {k: v.to(device) for k, v in event['batch'].items()}
    teacher = None
    if trainer.amortizer is not None:
        # Both branches apply the same MLP update on the spike batch.
        teacher = event['z'].to(device), event['solver']['norms'].to(device)
    step = trainer.prepare(batch, inverse=z.to(device=device, dtype=batch['Xt'].dtype))
    spike_loss = step['true_loss']
    change = trainer.update(batch, step, teacher=teacher, track_update=True)
    print(f'{label} spike update with loss {step["true_loss"]:.5g}, '
          f'grad norm {change["grad_norm"]:.5g}, update norm {change["parameter_update_norm"]:.5g}')
    rows = []
    for _ in range(steps):
        sync(trainer.device)
        started = time.perf_counter()
        batch = trainer.batch()
        step = trainer.prepare(batch)
        metrics = step_metrics(batch, step, trainer)
        updates = trainer.update(batch, step)
        sync(trainer.device)
        rows.append(dict(iteration=trainer.next_iteration - 1, **metrics, **updates,
                         t_total=time.perf_counter() - started))
    write_rows(path, rows)
    return dict(spike_loss=spike_loss,
                **change, average_inverse_steps=statistics.mean(r['inverse_steps'] for r in rows),
                average_inverse_func_evals=statistics.mean(r['inverse_func_evals'] for r in rows),
                average_iteration_seconds=statistics.mean(r['t_total'] for r in rows),
                median_true_loss=statistics.median(r['true_loss'] for r in rows))


def json_report(data):
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return json.dumps(clean(data), indent=2, allow_nan=False)


def spike(args):
    event = torch.load(args.capture, map_location='cpu', weights_only=False)
    state = event.get('state')
    batch = event['batch']
    weights = state['model'] if state is not None else event['model']
    width = weights['Wzs.0.weight'].shape[0]
    model = new_model(width, len(batch['t']), 'cpu', True).double()
    model.load_state_dict(weights)
    observed = event.get('z', batch.get('Z0'))
    z, summary, rows = refine_report(model, batch, observed, args.hessian_chunk)
    output = args.output or args.capture.with_suffix('')
    output.mkdir(exist_ok=True, parents=True)
    if rows:
        write_rows(output / 'samples.csv', rows)
    report = output / 'summary.json'
    report.write_text(json_report(summary) + '\n')
    print(json_report(summary))
    if args.follow:
        if state is None or observed is None or not observed.isfinite().all() or not math.isfinite(
            event['metrics']['loss']
        ):
            raise ValueError('Following both branches needs a finite cluster capture with Adam and RNG state')
        if summary['max_loss_vector_error_bound'] > 1e-4:
            raise ValueError('Reference inverse did not reach the error bound required for a controlled branch')
        summary['original_branch'] = follow_branch(event, observed, 'original', args.device, args.follow,
                                                    output / 'original.csv')
        summary['refined_branch'] = follow_branch(event, z, 'refined', args.device, args.follow,
                                                   output / 'refined.csv')
        report.write_text(json_report(summary) + '\n')
        print(json_report(summary))


def compare(args):
    profiles = {}
    for root in args.runs:
        with (root / 'profile.csv').open(newline='') as f:
            profiles[root] = list(csv.DictReader(f))
    horizon = min(len(rows) for rows in profiles.values())
    if args.iterations:
        if args.iterations > horizon:
            raise ValueError(f'Only {horizon} iterations are shared by every supplied run')
        horizon = args.iterations
    if not horizon:
        raise ValueError('All runs need at least one logged batch')
    common_eval = set(range(horizon))
    for rows in profiles.values():
        common_eval &= {i for i, r in enumerate(rows[:horizon]) if math.isfinite(float(r['heldout_sliced_w2']))}
    at = max(common_eval) if common_eval else None
    print(f'Comparing the first {horizon} completed batches')
    for root, rows in profiles.items():
        group = rows[:horizon]
        retries = sum(int(r['wolfe_retried']) for r in group)
        spikes = sum(float(r['spike_ratio']) >= args.spike_factor for r in group)
        total = float(group[-1]['train_seconds'])
        evals = statistics.mean(float(r['inverse_func_evals']) for r in group)
        peak = max(float(r['cuda_peak_reserved_mib']) for r in group)
        quality = f', held-out sliced W2 at {at} = {float(group[at]["heldout_sliced_w2"]):.5g}' if at is not None else ''
        print(f'{root.parent.name}/{root.name}: {total:.1f}s, {evals:.1f} closure evals/batch, '
              f'{spikes} spikes, {retries/horizon:.2%} retries, {peak:.0f} MiB peak reserved{quality}')


def evaluate(args):
    rows = []
    for path in args.checkpoints:
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = Config(**state['config'])
        model = new_model(config.width, args.size, 'cpu', True).double()
        model.load_state_dict(state['model'])
        batch = Batches(8191).sample(args.size, 'cpu')
        summary = refine_report(model, batch, None, args.hessian_chunk)[1]
        sw2 = sliced_w2(model.push_nograd(batch['X0'].double()), batch['X1'].double())
        rows.append(dict(checkpoint=str(path), method=config.method, iteration=state['next_iteration'],
                         heldout_sliced_w2=sw2, **summary))
    print(json_report(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=2)
    cmds = p.add_subparsers(dest='command', required=True)
    replay = cmds.add_parser('spike', help='Resolve a capture and optionally follow both branches')
    replay.add_argument('capture', type=Path)
    replay.add_argument('--output', type=Path)
    replay.add_argument('--follow', type=int, default=0, metavar='STEPS')
    replay.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='Device for the training branches')
    replay.add_argument('--hessian-chunk', type=int, default=128)
    summary = cmds.add_parser('compare', help='Compare profiles at a common iteration budget')
    summary.add_argument('runs', type=Path, nargs='+')
    summary.add_argument('--iterations', type=int)
    summary.add_argument('--spike-factor', type=float, default=4.)
    heldout = cmds.add_parser('evaluate', help='Evaluate checkpoints with a fixed full-range batch and float64 inverse')
    heldout.add_argument('checkpoints', type=Path, nargs='+')
    heldout.add_argument('--size', type=int, default=1024)
    heldout.add_argument('--hessian-chunk', type=int, default=128)
    args = p.parse_args()
    if args.threads < 1 or getattr(args, 'hessian_chunk', 1) < 1 or getattr(args, 'follow', 0) < 0:
        p.error('Use positive thread and Hessian chunk counts and nonnegative --follow')
    if getattr(args, 'size', 2) < 2 or (getattr(args, 'iterations', None) is not None and args.iterations < 1):
        p.error('Invalid evaluation size or comparison horizon')
    if getattr(args, 'follow', 0) and args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Pass --device cpu for a local branch replay.')
    torch.set_num_threads(args.threads)
    {'spike': spike, 'compare': compare, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
