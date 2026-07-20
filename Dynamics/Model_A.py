"""Model A dynamics: phi/x transforms and forward simulation via diffrax."""

import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt


def phi_to_x(phi):
    x = jnp.log(phi)
    return x


def x_to_phi(x):
    phi = jnp.exp(x)
    return phi


def dynamics_func_x_constant(t, x, args):
    (chi, mu, clamp) = args
    N = chi.shape[0]
    d = jnp.ones(N)
    if clamp is not None:
        d = d.at[:clamp].set(0.0)

    phi = jnp.exp(x)
    mu_x = 1 + jnp.log(phi) + jnp.matmul(chi, phi)
    mu_diff = mu_x + mu

    # constant-mobility multiplier: weight = d
    lam = jnp.sum(d * mu_diff) / jnp.sum(d)

    # phi-space flux with constant mobility, then convert to x via 1/phi
    dphi = -d * (mu_diff - lam)
    per_x = dphi * jnp.exp(-x)
    return per_x


def dynamics_func_x_dphi(t, x, args):
    (chi, mu, clamp) = args

    N = chi.shape[0]
    d = jnp.ones(N)
    if clamp is not None:
        d = d.at[:clamp].set(0.0)

    phi = jnp.exp(x)
    mu_x = 1 + jnp.log(phi) + jnp.matmul(chi, phi)
    mu_diff = mu_x + mu
    d_phi = jnp.multiply(d, phi)
    sum_d_phi = jnp.sum(d_phi)
    per_x = -(jnp.multiply(d, mu_diff) - jnp.multiply(d, jnp.sum(jnp.multiply(d_phi, mu_diff) / sum_d_phi)))
    return per_x


def forward_sim_x_ssolvent_clamp(phi0,
                                  chi,
                                  mu,
                                  clamp,
                                  t_end,
                                  dt,
                                  max_steps,
                                  samples=10,
                                  mobility="dphi"):
    """Forward-simulate the well-mixed (single-compartment, Model A) dynamics.

    Integrates the ODE in log-composition space (`x = log(phi)`) with
    diffrax's Dopri5 adaptive solver, so that `phi = exp(x)` stays strictly
    positive without an explicit simplex projection.

    Parameters
    ----------
    phi0 : (nc,) array
        Initial composition vector. Must be strictly positive (converted to
        log-space internally).
    chi : (nc, nc) array
        Flory-Huggins interaction matrix.
    mu : (nc,) array
        Chemical-potential offset / external field applied per component.
    clamp : int or None
        Number of leading components held fixed (zero mobility) -- e.g. to
        clamp "input" units in a classifier task. None means no clamping.
    t_end : float
        Total integration time. In the non-spatial (single-compartment)
        simulations reported in the paper, we used t_end=300.0.
    dt : float
        Initial step size passed to the Dopri5 adaptive solver (`dt0`). In
        the non-spatial simulations reported in the paper, we used dt=1e-1.
    max_steps : int
        Maximum number of solver steps; diffrax raises an error if this is
        exceeded before `t_end` is reached. In the non-spatial simulations
        reported in the paper, we used max_steps=30000.
    samples : int
        Number of time points saved, uniformly spaced between 0 and `t_end`.
    mobility : {"dphi", other}
        Selects the dynamics function: "dphi" uses phi-weighted mobility
        (`dynamics_func_x_dphi`); any other value uses constant mobility
        (`dynamics_func_x_constant`).

    Returns
    -------
    sol : diffrax.Solution
        Solution object; `x_to_phi(sol.ys[-1])` gives the final composition.
    """
    x0 = phi_to_x(phi0)
    if mobility == "dphi":
        term = ODETerm(dynamics_func_x_dphi)
    else:
        term = ODETerm(dynamics_func_x_constant)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.linspace(0, t_end, samples))

    sol = diffeqsolve(term,
                       solver,
                       t0=0,
                       t1=t_end,
                       dt0=dt,
                       y0=x0,
                       saveat=saveat,
                       args=(chi, mu, clamp),
                       max_steps=max_steps,
                       )

    return sol
