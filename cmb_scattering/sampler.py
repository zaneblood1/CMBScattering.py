"""Gibbs sampler for the Gaussian dust model of simulate_dust.py.

    d_freq = M_freq * B_freq * (f + a_freq * g) + n_freq

Each iteration of sample_gaussian_posterior

    1. draws the CMB field f from its conditional given the dust field and the parameters,
    2. draws the dust field g from its conditional given the CMB field and the parameters,
    3. updates dust_amplitude_ee, dust_amplitude_bb, dust_knee_ee, dust_knee_bb, r and
       frequency_exponent one at a time with a 1D random-walk Metropolis step on
       likelihoods.gaussian_posterior at the current fields.

The field draws use the Hoffman-Ribak algorithm: simulate a field and a noise realization
from their priors, then add the Wiener filter of the difference between the real and the
simulated data. Every operator of the model is diagonal in Fourier space, so the Wiener
filter is solved exactly mode by mode (wiener_filter) instead of by conjugate gradient.

Run from the terminal with, e.g.,

    python -m cmb_scattering.sampler --nside 64 --num_iterations 2000 --r_proposal_width 0.01

The dust temperature is held fixed throughout.
"""

import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from matplotlib.figure import Figure
from cmb_lensing.matrix_operators import get_identity_like, pinv
from cmb_scattering.likelihoods import FREQUENCY_EXPONENT_PRIOR_MEAN, gaussian_posterior
from cmb_scattering.simulate_dust import (DEFAULT_UK_ARCMIN, DUST_TEMPERATURE_K,
                                          FREQUENCIES_GHZ, PLANCK_DUST_KNEE_BB,
                                          PLANCK_DUST_KNEE_EE,
                                          PLANCK_DUST_BB_TO_EE_AMPLITUDE_RATIO,
                                          cmb_field_covariance_from_r,
                                          dust_covariance_operator_like,
                                          dust_frequency_scaling, draw_field, draw_noise,
                                          scale_dust_to_frequencies, simulate_data)

#the sampled parameters, in the order they are updated in each iteration
PARAM_ORDER = ["dust_amplitude_ee", "dust_amplitude_bb", "dust_knee_ee", "dust_knee_bb",
               "r", "frequency_exponent"]

#flat prior box of each parameter; a proposal outside it is rejected. The dust knees take
#BK18's prior range -1 < alpha_d < 0 (BICEP/Keck XIII, https://arxiv.org/abs/2110.00483);
#the other boxes are only wide enough to keep the covariances positive
PARAM_BOUNDS = {
    "dust_amplitude_ee": (0.0, 100.0),
    "dust_amplitude_bb": (0.0, 100.0),
    "dust_knee_ee": (-1.0, 0.0),
    "dust_knee_bb": (-1.0, 0.0),
    "r": (0.0, 1.0),
    "frequency_exponent": (0.0, 3.0),
}

#width of each parameter's Gaussian Metropolis proposal. First guesses, to be tuned by hand
#toward an acceptance rate near 44 %
DEFAULT_PROPOSAL_WIDTHS = {
    "dust_amplitude_ee": 0.2,
    "dust_amplitude_bb": 0.1,
    "dust_knee_ee": 0.02,
    "dust_knee_bb": 0.02,
    "r": 0.008,
    "frequency_exponent": 0.004,
}

#default truth of the simulated data in __main__. The BB dust amplitude is BK18's best fit
#A_d = 4.4 uK^2 (https://arxiv.org/abs/2110.00483) and the EE amplitude follows it through
#Planck 2018 XI's BB / EE ratio (https://arxiv.org/abs/1801.04945). Both dust slopes are
#Planck 2018 XI's, the frequency exponent is the centre of its BK18 prior, and r sits just
#under BK18's 95 % limit of 0.036
DEFAULT_DUST_AMPLITUDE_BB = 4.4
DEFAULT_R = 0.03
DEFAULT_TRUE_PARAMS = {
    "dust_amplitude_ee": DEFAULT_DUST_AMPLITUDE_BB / PLANCK_DUST_BB_TO_EE_AMPLITUDE_RATIO,
    "dust_amplitude_bb": DEFAULT_DUST_AMPLITUDE_BB,
    "dust_knee_ee": PLANCK_DUST_KNEE_EE,
    "dust_knee_bb": PLANCK_DUST_KNEE_BB,
    "r": DEFAULT_R,
    "frequency_exponent": FREQUENCY_EXPONENT_PRIOR_MEAN,
}

DEFAULT_NSIDE = 128
DEFAULT_THETA_PIX = 20.0
DEFAULT_MAP_SEED = 0
DEFAULT_CHAIN_SEED = 1
DEFAULT_NUM_ITERATIONS = 5000
#iterations between acceptance-rate printouts and updates of the log-posterior plot
DEFAULT_LOG_EVERY = 1
DEFAULT_OUT_DIR = "gaussian_sampler_output"
SAMPLE_HISTORIES_FILE = "sample_histories.npz"
LOGPOSTERIOR_PLOT_FILE = "logpdf_over_time.png"
LOGPOSTERIOR_FIGURE_SIZE = (10, 6)
#each sampled parameter's trace plot is saved as <name> + this suffix
TRACE_PLOT_FILE_SUFFIX = "_trace.png"
TRACE_FIGURE_SIZE = (10, 6)
PLOT_FORMAT = "png"
#a plot is written under its name + this suffix, then renamed into place
TEMPORARY_FILE_SUFFIX = ".tmp"

##############################################################
#FIELD DRAWS
##############################################################

def scale_by_frequency(field_or_operator, frequency_weights):
    """Multiply a field or operator by one weight per channel, adding (or keeping) the
    leading frequency axis."""
    weights = frequency_weights[:, None, None]
    updates = {name: weights * getattr(field_or_operator, name)
               for name in field_or_operator._matrix_names()}
    return field_or_operator.replace(**updates)

def sum_over_frequencies(field_or_operator):
    """Sum a field or operator over its leading frequency axis."""
    updates = {name: jnp.sum(getattr(field_or_operator, name), axis = 0)
               for name in field_or_operator._matrix_names()}
    return field_or_operator.replace(**updates)

def wiener_filter(data, prior_covariance, noise_covariance, mask, beam, frequency_weights):
    """Wiener filter of multi-frequency data for a field x with x ~ N(0, C) and
    data_freq = R_freq * x + n_freq, where R_freq = frequency_weights_freq * M * B.

    Returns C * (1 + C * sum_freq R N^-1 R)^-1 * sum_freq R N^-1 data_freq, which is the
    usual (C^-1 + sum_freq R N^-1 R)^-1 * sum_freq R N^-1 data_freq written so that modes
    with no prior power come out exactly zero. The mask and the beam are taken to be real
    and diagonal in Fourier space, so the solve is exact.
    """
    response = scale_by_frequency(mask * beam, frequency_weights)
    inverse_noise = pinv(noise_covariance)
    inverse_noise_power = sum_over_frequencies(response * inverse_noise * response)
    weighted_data = sum_over_frequencies(response * (inverse_noise * data))
    denominator = get_identity_like(prior_covariance) + prior_covariance * inverse_noise_power
    return pinv(denominator) * (prior_covariance * weighted_data)

@jax.jit
def gibbs_sample_field(data_residual, prior_covariance, noise_covariance, mask, beam,
                       frequency_weights, field_like, rng_key):
    """Draw a field x from its Gaussian conditional with the Hoffman-Ribak algorithm.

    `data_residual` is the data with every other component of the model removed, so that
    data_residual_freq = frequency_weights_freq * M * B * x + n_freq with x ~ N(0, C).
    The CMB has frequency_weights of one; the dust has the frequency scaling a_freq.
    """
    key_e, key_b, key_noise = jax.random.split(rng_key, 3)

    #simulate x' ~ N(0, C) and n' ~ N(0, C_n), and the data they would give
    simulated_field = draw_field(field_like, prior_covariance, key_e, key_b)
    simulated_noise = draw_noise(field_like, noise_covariance, key_noise)
    response = scale_by_frequency(mask * beam, frequency_weights)
    simulated_data = response * simulated_field + simulated_noise

    #x = x' + W * (d - d')
    correction = wiener_filter(data_residual - simulated_data, prior_covariance,
                               noise_covariance, mask, beam, frequency_weights)
    return simulated_field + correction

##############################################################
#PARAMETER UPDATES
##############################################################

@jax.jit
def eval_logposterior(params, cmb_field, dust_field, data, mask, beam, noise_covariance,
                      scalar_covariance, tensor_covariance, frequencies, dust_temperature):
    """gaussian_posterior at a {name: value} dict of the six parameters."""
    return gaussian_posterior(data, cmb_field, dust_field, mask, beam,
                              noise_covariance = noise_covariance,
                              scalar_covariance = scalar_covariance,
                              tensor_covariance = tensor_covariance,
                              frequencies = frequencies, dust_temperature = dust_temperature,
                              **params)

def metropolis_sample_parameter(logposterior_at, name, params, logposterior, bounds,
                                proposal_width, acceptance, rng_key):
    """One 1D random-walk Metropolis step on the parameter `name`.

    `logposterior_at(params)` evaluates the log-posterior at the current fields and
    `logposterior` is its value at `params`. Returns the (possibly updated) params and
    log-posterior, and appends whether the proposal was accepted to `acceptance[name]`.
    """
    key_proposal, key_accept = jax.random.split(rng_key)
    proposal = params[name] + proposal_width * float(jax.random.normal(key_proposal))

    accepted = False
    #reject proposals outside the flat prior box before evaluating the model
    low, high = bounds
    if low <= proposal <= high:
        proposed_params = {**params, name: proposal}
        proposed_logposterior = float(logposterior_at(proposed_params))
        #a non-finite log-posterior makes the comparison False, so the step is rejected
        accepted = float(jnp.log(jax.random.uniform(key_accept))) < proposed_logposterior - logposterior
        if accepted:
            params, logposterior = proposed_params, proposed_logposterior

    acceptance[name].append(accepted)
    return params, logposterior

##############################################################
#DIAGNOSTICS
##############################################################

def print_acceptance_rates(iteration, acceptance):
    rates = ", ".join(f"{name} = {np.mean(accepted):.2f}" for name, accepted in acceptance.items())
    print(f"iteration {iteration}: acceptance rates: {rates}", flush = True)

def save_figure(figure, path):
    #write the image to a temporary file and swap it into place in one step, so an image
    #viewer watching `path` never loads a half-written file (which shows up as flicker)
    temporary_path = path + TEMPORARY_FILE_SUFFIX
    figure.savefig(temporary_path, format = PLOT_FORMAT)
    os.replace(temporary_path, path)

def plot_logposterior(logposterior_values, path):
    """Save the log-posterior of every iteration so far as a figure at `path`."""
    figure = Figure(figsize = LOGPOSTERIOR_FIGURE_SIZE)
    axis = figure.subplots()
    axis.plot(logposterior_values)
    axis.set_title("Log-posterior over time")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Log-posterior value")
    save_figure(figure, path)

def plot_parameter_traces(sample_histories, names, out_dir, true_params = None):
    """Save one trace plot per parameter in `names`, <out_dir>/<name>_trace.png, from the
    samples drawn so far. `true_params` adds a line at each parameter's true value."""
    for name in names:
        figure = Figure(figsize = TRACE_FIGURE_SIZE)
        axis = figure.subplots()
        axis.plot(sample_histories[name], label = "chain")
        if true_params is not None:
            axis.axhline(true_params[name], color = "black", linestyle = "--", label = "truth")
            axis.legend()
        axis.set_title(f"{name} over time")
        axis.set_xlabel("Iteration")
        axis.set_ylabel(name)
        save_figure(figure, os.path.join(out_dir, name + TRACE_PLOT_FILE_SUFFIX))

##############################################################
#THE SAMPLER
##############################################################

def sample_gaussian_posterior(data, noise_covariance, scalar_covariance, tensor_covariance,
                              mask, beam, param_init,
                              proposal_widths = DEFAULT_PROPOSAL_WIDTHS,
                              param_bounds = PARAM_BOUNDS,
                              num_iterations = DEFAULT_NUM_ITERATIONS,
                              seed = DEFAULT_CHAIN_SEED, frequencies = FREQUENCIES_GHZ,
                              dust_temperature = DUST_TEMPERATURE_K,
                              log_every = DEFAULT_LOG_EVERY, out_dir = None,
                              true_params = None):
    """Gibbs-sample the joint posterior P(f, g, parameters | d) of the Gaussian dust model.

    Args:
        data: The multi-frequency data, a FlatS2 with a leading frequency axis (e.g.
            simulate_data's "data").
        noise_covariance: DiagonalEB with a leading frequency axis.
        scalar_covariance: DiagonalEB, with C_f(r) = scalar_covariance + r * tensor_covariance.
        tensor_covariance: DiagonalEB, the tensor covariance at r = 1.
        mask: DiagonalEB.
        beam: DiagonalEB.
        param_init: {name: value} starting point, all six parameters of PARAM_ORDER.
        proposal_widths: {name: width} of the Gaussian Metropolis proposal per parameter.
            The parameters are updated in this dict's order, and a parameter left out of it
            stays at its `param_init` value.
        param_bounds: {name: (low, high)}; proposals outside the range are rejected.
        num_iterations: Number of Gibbs iterations.
        seed: Seed of the chain's random draws.
        frequencies: The channels of the data, in GHz.
        dust_temperature: T_d of the dust modified blackbody, in K. Held fixed.
        log_every: Print the running acceptance rates, and update the log-posterior plot
            and the trace plots when `out_dir` is given, every this many iterations.
        out_dir: Directory of the log-posterior plot and of one trace plot per sampled
            parameter (<name>_trace.png). None skips the plots.
        true_params: Optional {name: value} of the true parameters, drawn as a reference
            line on the trace plots.

    Returns:
        (sample_histories, acceptance_rates, logposterior_values): sample_histories is
        {name: array of length num_iterations} for the six parameters, acceptance_rates is
        {name: fraction of accepted proposals} for the sampled ones, and
        logposterior_values is the log-posterior at the end of every iteration.
    """
    frequencies = jnp.asarray(frequencies, dtype = jnp.float64)
    cmb_frequency_weights = jnp.ones_like(frequencies)
    response = mask * beam

    #a single-channel zero field: the template of every draw and the starting dust field
    field_shape = data.polar_matrix_1.shape[-2:]
    zero_matrix = jnp.zeros(field_shape, dtype = data.polar_matrix_1.dtype)
    field_like = data.replace(polar_matrix_1 = zero_matrix, polar_matrix_2 = zero_matrix)
    dust_field = field_like

    params = {name: float(param_init[name]) for name in PARAM_ORDER}
    sample_histories = {name: [] for name in PARAM_ORDER}
    acceptance = {name: [] for name in proposal_widths}
    logposterior_values = []
    rng_key = jax.random.PRNGKey(seed)

    for iteration in range(1, num_iterations + 1):
        rng_key, key_cmb, key_dust = jax.random.split(rng_key, 3)

        #the covariances and the dust frequency scaling at the current parameters
        field_covariance = cmb_field_covariance_from_r(params["r"], scalar_covariance,
                                                       tensor_covariance)
        dust_covariance = dust_covariance_operator_like(params["dust_amplitude_ee"],
                                                        params["dust_knee_ee"],
                                                        params["dust_amplitude_bb"],
                                                        params["dust_knee_bb"],
                                                        scalar_covariance)
        frequency_scaling = dust_frequency_scaling(frequencies, params["frequency_exponent"],
                                                   dust_temperature)

        #1. draw the CMB field given the dust field
        dust_data = response * scale_dust_to_frequencies(dust_field, frequency_scaling)
        cmb_field = gibbs_sample_field(data - dust_data, field_covariance, noise_covariance,
                                       mask, beam, cmb_frequency_weights, field_like, key_cmb)

        #2. draw the dust field given the CMB field
        cmb_data = response * cmb_field
        dust_field = gibbs_sample_field(data - cmb_data, dust_covariance, noise_covariance,
                                        mask, beam, frequency_scaling, field_like, key_dust)

        #3. update the parameters one at a time at the new fields
        def logposterior_at(proposed_params):
            return eval_logposterior(proposed_params, cmb_field, dust_field, data, mask, beam,
                                     noise_covariance, scalar_covariance, tensor_covariance,
                                     frequencies, dust_temperature)

        logposterior = float(logposterior_at(params))
        for name, proposal_width in proposal_widths.items():
            rng_key, key_param = jax.random.split(rng_key)
            params, logposterior = metropolis_sample_parameter(logposterior_at, name, params,
                                                               logposterior, param_bounds[name],
                                                               proposal_width, acceptance,
                                                               key_param)

        for name in PARAM_ORDER:
            sample_histories[name].append(params[name])
        logposterior_values.append(logposterior)

        #4. diagnostics for hand-tuning the proposal widths
        if iteration % log_every == 0:
            print_acceptance_rates(iteration, acceptance)
            if out_dir is not None:
                plot_logposterior(logposterior_values, os.path.join(out_dir, LOGPOSTERIOR_PLOT_FILE))
                plot_parameter_traces(sample_histories, proposal_widths, out_dir, true_params)

    sample_histories = {name: np.asarray(values) for name, values in sample_histories.items()}
    acceptance_rates = {name: float(np.mean(accepted)) for name, accepted in acceptance.items()}
    return sample_histories, acceptance_rates, np.asarray(logposterior_values)

##############################################################
#COMMAND LINE
##############################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description = "Simulate multi-frequency E + B data with Gaussian dust and "
                      "Gibbs-sample r and the dust parameters from it.")
    parser.add_argument("--nside", type = int, default = DEFAULT_NSIDE,
                        help = "number of pixels on a side")
    parser.add_argument("--theta_pix", type = float, default = DEFAULT_THETA_PIX,
                        help = "pixel width in arcmin")
    parser.add_argument("--uk_arcmin", type = float, default = DEFAULT_UK_ARCMIN,
                        help = "polarization white noise level of every channel, in uK-arcmin")
    parser.add_argument("--dust_temperature", type = float, default = DUST_TEMPERATURE_K,
                        help = "dust temperature in K (held fixed)")
    parser.add_argument("--map_seed", type = int, default = DEFAULT_MAP_SEED,
                        help = "seed of the simulated data")
    parser.add_argument("--chain_seed", type = int, default = DEFAULT_CHAIN_SEED,
                        help = "seed of the sampler's random draws")
    parser.add_argument("--num_iterations", type = int, default = DEFAULT_NUM_ITERATIONS,
                        help = "number of Gibbs iterations")
    parser.add_argument("--log_every", type = int, default = DEFAULT_LOG_EVERY,
                        help = "iterations between acceptance-rate printouts and plot updates")
    parser.add_argument("--out_dir", type = str, default = DEFAULT_OUT_DIR,
                        help = "directory the sample histories and the log-posterior plot go to")
    for name in PARAM_ORDER:
        parser.add_argument(f"--{name}", type = float, default = DEFAULT_TRUE_PARAMS[name],
                            help = f"true {name} of the simulated data")
        parser.add_argument(f"--{name}_init", type = float, default = None,
                            help = f"starting {name} of the chain (default: the true value)")
        parser.add_argument(f"--{name}_proposal_width", type = float,
                            default = DEFAULT_PROPOSAL_WIDTHS[name],
                            help = f"width of the Metropolis proposal of {name}")
    return parser.parse_args()

if __name__ == "__main__":

    arguments = vars(parse_arguments())
    os.makedirs(arguments["out_dir"], exist_ok = True)

    #the truth of the simulated data, the chain's starting point and its proposal widths
    true_params = {name: arguments[name] for name in PARAM_ORDER}
    param_init = {name: true_params[name] if arguments[f"{name}_init"] is None
                  else arguments[f"{name}_init"] for name in PARAM_ORDER}
    proposal_widths = {name: arguments[f"{name}_proposal_width"] for name in PARAM_ORDER}

    #generate a "ground truth" simulated data set
    sim = simulate_data(arguments["nside"], arguments["theta_pix"], arguments["map_seed"],
                        uk_arcmin = arguments["uk_arcmin"],
                        dust_temperature = arguments["dust_temperature"], **true_params)

    #run the sampling algorithm
    sample_histories, acceptance_rates, logposterior_values = sample_gaussian_posterior(
        sim["data"], sim["noise_covariance"], sim["scalar_covariance"],
        sim["tensor_covariance"], sim["mask"], sim["beam"], param_init,
        proposal_widths = proposal_widths, num_iterations = arguments["num_iterations"],
        seed = arguments["chain_seed"], frequencies = sim["frequencies"],
        dust_temperature = arguments["dust_temperature"],
        log_every = arguments["log_every"], out_dir = arguments["out_dir"],
        true_params = true_params)

    plot_logposterior(logposterior_values,
                      os.path.join(arguments["out_dir"], LOGPOSTERIOR_PLOT_FILE))
    plot_parameter_traces(sample_histories, proposal_widths, arguments["out_dir"], true_params)
    np.savez(os.path.join(arguments["out_dir"], SAMPLE_HISTORIES_FILE),
             logposterior = logposterior_values, **sample_histories)
    for name in PARAM_ORDER:
        print(f"{name}: truth = {true_params[name]:.4g}, "
              f"mean = {np.mean(sample_histories[name]):.4g}, "
              f"std = {np.std(sample_histories[name]):.4g}, "
              f"acceptance rate = {acceptance_rates[name]:.2f}")
