import numpy as np
import torch
from tqdm import tqdm
from .icnn import GradNNGeneric
from IPython.display import clear_output
import torch.nn.functional as F
from .tools import freeze_context
from typing import Optional, Dict, Type
from dataclasses import dataclass, field, replace
import time

###################
# ORACLE
###########

#TODO

###################
# PARAMETERS
############

@dataclass
class InvMethodParamsGeneric:

    @property
    def name(self):
        raise NotImplementedError()

    verbose: bool = False
    report_final_norms: bool = False

@dataclass
class TorchoptimInvParams(InvMethodParamsGeneric):

    @property
    def name(self):
        return 'torchoptim'

    optim_cls: Type[torch.optim.Optimizer] = torch.optim.Adam # optimizer class to use
    lr: float = 0.1 # learning rate of Adam optimizer
    max_iter: int = 1000 # max iteration steps
    grad_tol: float = 1e-5 # (batch-averaged) grad norm stopping criteria
    grad_tol_max: Optional[float] = 1e-5 # (batch-max) grad norm stopping criteria
    t_min: float = 1e-5 # threshold to prevent inverting X_t with t < t_min
    optim_params: Dict = field(default_factory=lambda: {}) # additional parameters of optimizer
    init_method: str = 'Xt'

@dataclass
class TorchlbfgsInvParams(InvMethodParamsGeneric):

    @property
    def name(self):
        return 'torchlbfgs'
    
    lbfgs_params: Dict = field(default_factory= lambda: {
        'tolerance_grad': 1e-4
    }) # additional parameters of LBFGS optimizer
    t_min: float = 1e-5 # threshold to prevent inverting X_t with t < t_min
    init_method: str = 'Xt'
    max_iter: int = 50
    record_steps: bool = False # record the residual after every outer step (costs one extra grad eval per step)
    initial_guess: Optional[torch.Tensor] = None  # one starting point per row of Xt
    wolfe_retry_displacement: Optional[float] = None  # retry if any returned point is farther from Xt
    report_solution: bool = False  # add the detached full-batch result to stats

@dataclass
class ManualInvParams(InvMethodParamsGeneric):
    
    @property
    def name(self):
        return 'manual'

    double_precision: bool = False
    max_iter: int = 1000
    grad_tol: float = 1e-5
    no_progress_stop: int = 5
    no_progress_tol: float = 1e-8
    min_lambda: float = 1e-8
    lr_base: float = 1.0
    lambda_corrector: float = 1e-6

###################
# functions
###############

def FUNC(D, Z0, Xt, t):
    return t * D(Z0) - (Z0 * Xt).sum(-1).view(-1, 1) + (1. - t) * (Z0 * Z0).sum(-1).view(-1, 1) / 2.

def FUNC_grad(D, Z0, Xt, t):
    return t * D.push_nograd(Z0) - Xt + (1. - t) * Z0

def FUNC_grad_norm(D, Z0, Xt, t):
    return torch.norm(FUNC_grad(D, Z0, Xt, t).detach(), dim=-1)

###################
# SOLVERS
###########

def ofm_inverse_manual(
    D: GradNNGeneric,
    Xt: torch.Tensor,
    t: torch.Tensor,
    params: ManualInvParams,
    stats: Dict = dict()
):  
    assert t.size(0) == Xt.size(0)
    assert len(t.shape) == 2
    assert t.size(1) == 1
    
    t_strt = time.perf_counter()

    if params.double_precision:
        D = D.double()
    
    t = t.clone()
    Xs_curr = Xt.clone()
    if params.double_precision:
        Xs_curr = Xs_curr.double()
        t = t.double()
    Xs_prev = Xs_curr.clone()
    Xs_prev.detach_()
    lr_base = params.lr_base
    j = 0
    max_grad_norm_history = []
    mask = torch.arange(0, Xs_curr.size(0), dtype=int)
    while True:
        _Xs_prev = Xs_prev[mask] # we optimize
        _Xs_prev.requires_grad_(True) 
        _Xs_curr = Xs_curr[mask] # reference
        _t = t[mask]
        _grad = FUNC_grad(D, _Xs_prev, _Xs_curr, _t).detach()
        prev_grad_norms = torch.norm(_grad, dim=-1)
        _lambdas = (lr_base / torch.sqrt(prev_grad_norms + params.lambda_corrector)).view(-1, 1)
        while True:
            _new_Xs = _Xs_prev - _lambdas * _grad
            curr_grad_norms = FUNC_grad_norm(D, _new_Xs, _Xs_curr, _t)
            diff = prev_grad_norms - curr_grad_norms
            if torch.sum(diff <= 0.) == 0:
                break
            if torch.min(_lambdas) < params.min_lambda:
                break
            _lambdas[diff <= 0.] *= 0.5

        _Xs_prev = _Xs_prev - _lambdas * _grad
        final_grad_norms = FUNC_grad_norm(D, _Xs_prev, _Xs_curr, _t)
        max_grad_norm_history.append(final_grad_norms.max().item())
        acheve_mask = (final_grad_norms < params.grad_tol).cpu()
        Xs_prev[mask] = _Xs_prev.detach()
        mask = mask[~acheve_mask]
        # print(len(mask))
        if len(mask) == 0:
            if params.verbose:
                print('--------------------------')
                print('Inverse Manual summary')
                print('n_iters: {}'.format(j))
                print('max grad diff: ', FUNC_grad_norm(D, Xs_prev, Xs_curr, t).max().item())
                print('--------------------------')
            break
        if j > params.max_iter:
            if params.verbose:
                print('--------------------------')
                print('Inverse Manual summary')
                print('n_iters: {}'.format(j))
                print('stopped since max_iter acheved')
                print('N not converged: ', len(mask))
                print('max grad diff: ', FUNC_grad_norm(D, Xs_prev, Xs_curr, t).max().item())
                print('--------------------------')
            break
        if j > params.no_progress_stop:
            if np.max(max_grad_norm_history[-params.no_progress_stop:]) - np.min(max_grad_norm_history[-params.no_progress_stop:]) < params.no_progress_tol:
                if params.verbose:
                    print('--------------------------')
                    print('Inverse Manual summary')
                    print('n_iters: {}'.format(j))
                    print('stopped since no progress acheved')
                    print('N not acheved: ', len(mask))
                    print('max grad diff: ', FUNC_grad_norm(D, Xs_prev, Xs_curr, t).max().item())
                    print('--------------------------')
                break
        j += 1
    stats['time'] = time.perf_counter() - t_strt
    stats['steps'] = j
    stats['converged'] = float(len(Xs_prev) - len(mask))/float(len(Xs_prev))
    fin_norms = FUNC_grad_norm(D, Xs_prev, Xs_curr, t)
    if params.report_final_norms:
        stats['norms'] = fin_norms
    stats['norms_max'] = fin_norms.max().item()
    stats['norms_mean'] = fin_norms.mean().item()
    Xs_prev = Xs_prev.detach()
    if params.double_precision:
        Xs_prev = Xs_prev.float()
        D = D.float()
    return Xs_prev

def ofm_inverse_torchlbfgs(
        model: GradNNGeneric,
        Xt: torch.Tensor,
        t: torch.Tensor,
        params: TorchlbfgsInvParams,
        stats: Optional[Dict] = None,
):
    assert t.shape == (Xt.size(0), 1)
    if stats is None:
        stats = {}
    if params.initial_guess is not None and params.initial_guess.shape != Xt.shape:
        raise ValueError("initial_guess must have the same shape as Xt")
    if params.wolfe_retry_displacement is not None and len(Xt):
        return _inverse_with_wolfe_retry(model, Xt, t, params, stats)

    Y_inv_final = Xt.clone()
    indices = t[:, 0] >= params.t_min
    Xt = Y_inv_final[indices]
    t = t[indices]
    if params.initial_guess is not None:
        Y_inv = params.initial_guess.detach().to(Xt)[indices]
    elif params.init_method == 'Xt':
        Y_inv = Xt.clone()
    elif params.init_method == 'normal':
        Y_inv = torch.randn_like(Xt)
    elif params.init_method == 'zero':
        Y_inv = torch.zeros_like(Xt)
    elif params.init_method == 'uniform':
        Y_inv = torch.rand_like(Xt)
    else:
        raise Exception(f"Unknown init_method: '{params.init_method}'")
    Y_inv.requires_grad_(True)

    stats['n_eligible'] = len(Xt)
    if not len(Xt):
        stats.update(time=0., time_diag=0., steps=0, norms_max=0., norms_mean=0.,
                     func_evals=0, inner_iterations=0)
        if params.record_steps:
            stats.update(step_stats=np.zeros((1, 3), dtype=np.float32),
                         step_n_iter=[0], step_func_evals=[0])
        if params.report_final_norms:
            stats['norms'] = Xt.new_empty((0,))
        if params.report_solution:
            stats['solution'] = Y_inv_final.detach()
        return Y_inv_final.detach()

    opt = torch.optim.LBFGS([Y_inv], **params.lbfgs_params)
    t_strt = time.perf_counter()
    n_steps = 0
    prev_n_iter = 0

    def closure():
        opt.zero_grad()
        loss = (t * model(Y_inv)).sum() - (Xt * Y_inv).sum() + ((1. - t)/2. * Y_inv * Y_inv).sum()
        loss.backward()
        return loss

    # per-step convergence diagnostics; their own time is measured and excluded from stats['time']
    step_stats, step_n_iter, step_func_evals = [], [], []
    t_diag = 0.

    def record_step():
        nonlocal t_diag
        if Xt.is_cuda:
            torch.cuda.synchronize()
        t_rec = time.perf_counter()
        # gradient at the current iterate (the optimizer's prev_flat_grad lags one iteration behind)
        loss = (t * model(Y_inv)).sum() - (Xt * Y_inv).sum() + ((1. - t)/2. * Y_inv * Y_inv).sum()
        g = torch.autograd.grad(loss, Y_inv)[0].detach()
        g_norms = torch.norm(g.flatten(start_dim=1), p=2, dim=1)
        # max |g| is what tolerance_grad is compared against, i.e. a criterion over the whole batch
        step_stats.append(torch.stack([g.abs().max(), g_norms.max(), g_norms.mean()]))
        opt_state = opt.state_dict()['state'].get(0, dict(n_iter=0, func_evals=0))
        step_n_iter.append(opt_state['n_iter'])
        step_func_evals.append(opt_state['func_evals'])
        if Xt.is_cuda:
            torch.cuda.synchronize()
        t_diag += time.perf_counter() - t_rec

    if params.record_steps:
        record_step()  # residual at the initial point
    for _ in range(params.max_iter):
        opt.step(closure)
        n_iter = opt.state[Y_inv]['n_iter']
        n_steps += 1
        if params.record_steps:
            record_step()
        if n_iter == prev_n_iter:
            break
        prev_n_iter = n_iter
    if params.record_steps:
        t_rec = time.perf_counter()
        stats['step_stats'] = torch.stack(step_stats).cpu().numpy() # (n_steps + 1, 3): max|g|, max l2 norm, mean l2 norm
        stats['step_n_iter'] = step_n_iter
        stats['step_func_evals'] = step_func_evals
        t_diag += time.perf_counter() - t_rec
    t_elaps = time.perf_counter() - t_strt - t_diag
    stats['time_diag'] = t_diag
    norms = FUNC_grad_norm(model, Y_inv, Xt, t).detach()
    stats['time'] = t_elaps
    if params.report_final_norms:
        stats['norms'] = norms
    stats['norms_max'] = norms.max().item()
    stats['norms_mean'] = norms.mean().item()
    stats['steps'] = n_steps
    stats['func_evals'] = opt.state[Y_inv].get('func_evals', 0)
    stats['inner_iterations'] = opt.state[Y_inv].get('n_iter', 0)
    if params.verbose:
        print('--------------------------')
        print('Inverse TorchLBFGS summary')
        print('Optimizer: {}'.format(opt.__class__.__name__))
        print('Batch size: {}'.format(Xt.size(0)))
        print('Grad norms max: {}'.format(norms.max().item()))
        print('Grad norms av: {}'.format(norms.mean().item()))
        print(f'Iteration steps: {n_steps}')
        print(f'Time elapsed (s): {t_elaps}')
        print('--------------------------')
    Y_inv_final[indices] = Y_inv.detach()
    if params.report_solution:
        stats['solution'] = Y_inv_final.detach()
    return Y_inv_final.detach()


def _inverse_with_wolfe_retry(model, Xt, t, params, stats):
    """Try plain L-BFGS and restart the batch if its inverse points make an extreme excursion."""
    started = time.perf_counter()
    limit = params.wolfe_retry_displacement
    if not 0. < limit < float('inf'):
        raise ValueError("wolfe_retry_displacement must be positive and finite")
    plain_params = replace(
        params, wolfe_retry_displacement=None,
        lbfgs_params={**params.lbfgs_params, 'line_search_fn': None},
    )
    plain_stats = {}
    solution = ofm_inverse_torchlbfgs(model, Xt, t, plain_params, plain_stats)
    displacement = (solution - Xt).norm(dim=1)
    max_displacement = torch.where(displacement.isfinite(), displacement, torch.inf).max().item()
    retry_stats = {}
    retried = max_displacement > limit
    if retried:
        # The plain attempt shared one optimizer across the batch. Restart
        # every row from Xt to discard its optimizer state and any bad warm start.
        retry_params = replace(
            plain_params, initial_guess=None, init_method='Xt',
            lbfgs_params={**plain_params.lbfgs_params, 'line_search_fn': 'strong_wolfe'},
        )
        solution = ofm_inverse_torchlbfgs(model, Xt, t, retry_params, retry_stats)
        if not solution.isfinite().all().item():
            raise FloatingPointError("Strong Wolfe retry returned a non-finite inverse")

    # The curve and residuals describe the returned solution. Keep separate
    # counters to account for both attempts, including any discarded work.
    stats.update(retry_stats if retried else plain_stats)
    stats.update(
        wolfe_retried=retried, plain_max_displacement=max_displacement,
        plain_steps=plain_stats['steps'], wolfe_steps=retry_stats.get('steps', 0),
        time_diag=plain_stats['time_diag'] + retry_stats.get('time_diag', 0.),
    )
    stats['plain_func_evals'] = plain_stats['func_evals']
    stats['wolfe_func_evals'] = retry_stats.get('func_evals', 0)
    stats['func_evals'] = stats['plain_func_evals'] + stats['wolfe_func_evals']
    stats['inner_iterations'] = plain_stats['inner_iterations'] + retry_stats.get('inner_iterations', 0)
    stats['time'] = time.perf_counter() - started - stats['time_diag']
    return solution


def ofm_inverse_torchoptim(
        model: GradNNGeneric,
        Xt: torch.Tensor,
        t: torch.Tensor,
        params: TorchoptimInvParams,
        stats: Dict = dict()
):
    assert t.size(0) == Xt.size(0)
    assert len(t.shape) == 2
    assert t.size(1) == 1

    # freeze(model)
    Y_inv_final = Xt.clone()
    indices = t.squeeze() >= params.t_min
    Xt = Y_inv_final[indices]
    t = t[indices]
    if params.init_method == 'Xt':
        Y_inv = Xt.clone()
    elif params.init_method == 'normal':
        Y_inv = torch.randn_like(Xt)
    elif params.init_method == 'zero':
        Y_inv = torch.zeros_like(Xt)
    elif params.init_method == 'uniform':
        Y_inv = torch.rand_like(Xt)
    else:
        raise Exception(f"Unknown init_method: '{params.init_method}'")
    Y_inv.requires_grad_(True)

    # optimizer

    opt = params.optim_cls([Y_inv], lr=params.lr, **params.optim_params)

    # opt = torch.optim.SGD([Y_inv], lr=params.lr)

    t_strt = time.perf_counter()
    n_steps = 0
    norms = None
    if len(Xt) > 0:
        for _ in range(params.max_iter):
            loss = (t * model(Y_inv)).sum() - (Xt * Y_inv).sum() + ((1. - t)/2. * Y_inv * Y_inv).sum()
            loss.backward()
            opt.step()

            norms = torch.norm(Y_inv.grad.data.flatten(start_dim=1), p=2, dim=1)
            stop_iter_flag = False
            if params.grad_tol_max is None:
                stop_iter_flag =  norms.mean() < params.grad_tol
            else:
                stop_iter_flag = norms.max() < params.grad_tol_max
            opt.zero_grad()
            n_steps += 1
            if stop_iter_flag:
                break
    t_elaps = time.perf_counter() - t_strt
    stats['time'] = t_elaps
    if params.report_final_norms:
        stats['norms'] = norms
    stats['norms_max'] = norms.max().item()
    stats['norms_mean'] = norms.mean().item()
    stats['steps'] = n_steps
    if params.verbose:
        print('--------------------------')
        print('Inverse TorchOptim summary')
        print('Optimizer: {}'.format(opt.__class__.__name__))
        print('Batch size: {}'.format(Xt.size(0)))
        if params.grad_tol_max is None:
            print('Average tolerance: {}'.format(params.grad_tol))
        else:
            print('Min tolerance: {}'.format(params.grad_tol_max))
        print(f'Iteration steps: {n_steps}')
        print(f'Time elapsed (s): {t_elaps}')
        print('--------------------------')
    Y_inv_final[indices] = Y_inv.detach()
    return Y_inv_final.detach()


################
# main function

INV_METHODS2SOLVERS = {
    'torchoptim': ofm_inverse_torchoptim,
    'torchlbfgs': ofm_inverse_torchlbfgs,
    'manual': ofm_inverse_manual
}

INV_METHODS2PARAMCLASS = {
    'torchoptim': TorchoptimInvParams,
    'torchlbfgs': TorchlbfgsInvParams,
    'manual': ManualInvParams
}

def ofm_inverse(
        model: GradNNGeneric,
        Xt: torch.Tensor,
        t: torch.Tensor,
        method_params: InvMethodParamsGeneric = TorchoptimInvParams(),
        stats: Optional[Dict] = None,
):
    if stats is None:
        stats = {}
    if len(t.shape) == 1:
        t = t.unsqueeze(1)
    assert t.size(0) == Xt.size(0)
    t = t.to(Xt)
    assert method_params.name in INV_METHODS2SOLVERS.keys()
    # assert isinstance(method_params, INV_METHODS2PARAMCLASS[method_params.name])
    solver = INV_METHODS2SOLVERS[method_params.name]
    with freeze_context(model):
        return solver(model, Xt, t, method_params, stats)

def ofm_inverse_grad_norms(
    model: GradNNGeneric,
    Z0: torch.Tensor,
    Xt: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if len(t.shape) == 1:
        t = t.unsqueeze(1)
    assert t.size(0) == Xt.size(0)
    t = t.to(Xt)
    with freeze_context(model):
        return FUNC_grad_norm(model, Z0, Xt, t).detach()