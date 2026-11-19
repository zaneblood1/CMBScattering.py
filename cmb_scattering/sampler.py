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
simulated data. With a mask and a beam that are diagonal in Fourier space (the default
identity, or any DiagonalEB) every operator of the model is, and the Wiener filter is solved
exactly mode by mode (wiener_filter). A pixel-space mask (instrument.PixelMask) is not, and
the Wiener filter is then solved by conjugate gradient (conjugate_gradient_wiener_filter).

Run from the terminal with, e.g.,

    python -m cmb_scattering.sampler --nside 64 --num_iterations 2000 --r_proposal_width 0.01

sample_scattering_posterior samples the non-Gaussian model of
likelihoods.scattering_posterior instead, in which the dust prior is an L2 loss on the
log scattering coefficients of g. The CMB draw is unchanged, the dust field is no longer
Gaussian given the rest and is updated with a Hamiltonian Monte Carlo step
(hmc_sample_dust), and only r and frequency_exponent are left as parameters. With
gaussian_dust_prior = True the Gaussian dust prior is kept next to the scattering loss and
the four dust amplitudes and knees are parameters again.

The dust temperature is held fixed throughout.
"""

import argparse
from functools import partial
import os
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import jax.scipy.sparse.linalg
import numpy as np
jax.config.update("jax_enable_x64", True)
from matplotlib.figure import Figure
from cmb_lensing.matrix_operators import DiagonalEB, get_identity_like, pinv
from cmb_scattering.likelihoods import (DEFAULT_SCATTERING_WIDTH, FREQUENCY_EXPONENT_PRIOR_MEAN,
                                        gaussian_posterior, scattering_posterior)
from cmb_scattering.simulate_dust import (DEFAULT_PYSM_PRESET, DEFAULT_UK_ARCMIN,
                                          DUST_MODELS, DUST_TEMPERATURE_K, GAUSSIAN_DUST,
                                          FREQUENCIES_GHZ, PLANCK_DUST_KNEE_BB,
                                          PLANCK_DUST_KNEE_EE,
                                          PLANCK_DUST_BB_TO_EE_AMPLITUDE_RATIO,
                                          cmb_field_covariance_from_r,
                                          dust_covariance_operator_like,
                                          dust_frequency_scaling, draw_field, draw_noise,
                                          simulate_data)

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

#the parameters of the scattering transform posterior, which has no dust amplitudes or knees
SCATTERING_PARAM_ORDER = ["r", "frequency_exponent"]
#Metropolis proposal widths of sample_scattering_posterior, hand-tuned on PySM d9 dust
DEFAULT_SCATTERING_PROPOSAL_WIDTHS = {
    "r": 0.008,
    "frequency_exponent": 0.002,
}
#leapfrog steps and step size of the dust field's HMC update. The mass matrix is the
#data's precision on the dust, so a step size of one is one natural time unit of every mode
DEFAULT_HMC_NUM_STEPS = 10
DEFAULT_HMC_STEP_SIZE = 0.15
#relative residual at which the conjugate-gradient Wiener filter of a pixel mask stops, and
#its cap on the number of iterations
CONJUGATE_GRADIENT_TOLERANCE = 1e-6
CONJUGATE_GRADIENT_MAX_ITERATIONS = 500
#the name the dust field's HMC acceptance rate is reported under
DUST_FIELD_NAME = "dust_field"

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

#the parameters that are still inputs of the simulation, and so have a true value, when the
#dust comes from PySM
PYSM_TRUE_PARAMS = ["r", "frequency_exponent"]

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

def apply_response(field, mask, beam, frequency_weights):
    """R * field with R_freq = frequency_weights_freq * M * B: a single-channel field as it
    enters the data of every channel (leading frequency axis)."""
    return mask * (beam * scale_by_frequency(field, frequency_weights))

def diagonal_response(mask, beam):
    """M * B as a DiagonalEB. A pixel mask is not diagonal in Fourier space and is replaced
    by its rms, which is only good enough for a preconditioner, an HMC mass matrix or a
    starting point."""
    if isinstance(mask, DiagonalEB):
        return mask * beam
    return beam * jnp.sqrt(jnp.mean(mask.pixel_map**2))

def inverse_noise_power(noise_covariance, mask, beam, frequency_weights):
    """sum_freq R N^-1 R with R_freq = frequency_weights_freq * M * B: the precision the
    data alone puts on a field that enters channel `freq` through R_freq. Exact for a
    Fourier-diagonal mask, an approximation for a pixel mask (diagonal_response)."""
    response = scale_by_frequency(diagonal_response(mask, beam), frequency_weights)
    return sum_over_frequencies(response * pinv(noise_covariance) * response)

def inverse_noise_weighted_data(data, noise_covariance, mask, beam, frequency_weights):
    """sum_freq R N^-1 data_freq with R_freq = frequency_weights_freq * M * B, with the same
    approximation for a pixel mask as inverse_noise_power."""
    response = scale_by_frequency(diagonal_response(mask, beam), frequency_weights)
    return sum_over_frequencies(response * (pinv(noise_covariance) * data))

def conjugate_gradient_wiener_filter(data, prior_covariance, noise_covariance, mask, beam,
                                     frequency_weights):
    """wiener_filter for a mask that is not diagonal in Fourier space, by preconditioned
    conjugate gradient on the real-space maps.

    Solves (1 + C^1/2 R^T N^-1 R C^1/2) y = C^1/2 R^T N^-1 data and returns C^1/2 y, the
    form in which modes with no prior power come out exactly zero. R^T is the transpose of
    the response, taken by JAX. The preconditioner is the exact solution for the
    Fourier-diagonal stand-in of the mask.
    """
    field_shape = data.polar_matrix_1.shape[-2:]
    zero_matrix = jnp.zeros(field_shape, dtype = data.polar_matrix_1.dtype)
    field_like = data.replace(polar_matrix_1 = zero_matrix, polar_matrix_2 = zero_matrix)
    sqrt_covariance = prior_covariance.replace(matrix_EE = jnp.sqrt(prior_covariance.matrix_EE),
                                               matrix_BB = jnp.sqrt(prior_covariance.matrix_BB))
    inverse_noise = pinv(noise_covariance)

    def scale_by_sqrt_covariance(maps):
        return field_to_maps(sqrt_covariance * maps_to_field(maps, field_like))

    def response(maps):
        return field_to_maps(apply_response(maps_to_field(maps, field_like), mask, beam,
                                            frequency_weights))

    def response_transpose(data_maps):
        example_input = field_to_maps(field_like)                          
        transpose_function = jax.linear_transpose(response, example_input) 
        (result,) = transpose_function(data_maps)                          
        return result
    
    def weight_by_inverse_noise(data_maps):
        return field_to_maps(inverse_noise * maps_to_field(data_maps, data))

    def system(maps):
        data_maps = weight_by_inverse_noise(response(scale_by_sqrt_covariance(maps)))
        return maps + scale_by_sqrt_covariance(response_transpose(data_maps))

    data_precision = inverse_noise_power(noise_covariance, mask, beam, frequency_weights)
    inverse_diagonal = pinv(get_identity_like(prior_covariance) + prior_covariance * data_precision)

    def preconditioner(maps):
        return field_to_maps(inverse_diagonal * maps_to_field(maps, field_like))

    right_hand_side = scale_by_sqrt_covariance(
                      response_transpose(weight_by_inverse_noise(field_to_maps(data))))
    solution, _ = jax.scipy.sparse.linalg.cg(system, right_hand_side, M = preconditioner,
                                             tol = CONJUGATE_GRADIENT_TOLERANCE,
                                             maxiter = CONJUGATE_GRADIENT_MAX_ITERATIONS)
    return maps_to_field(scale_by_sqrt_covariance(solution), field_like)

def wiener_filter(data, prior_covariance, noise_covariance, mask, beam, frequency_weights):
    """Wiener filter of multi-frequency data for a field x with x ~ N(0, C) and
    data_freq = R_freq * x + n_freq, where R_freq = frequency_weights_freq * M * B.

    Returns C * (1 + C * sum_freq R N^-1 R)^-1 * sum_freq R N^-1 data_freq, which is the
    usual (C^-1 + sum_freq R N^-1 R)^-1 * sum_freq R N^-1 data_freq written so that modes
    with no prior power come out exactly zero. For a mask and a beam that are real and
    diagonal in Fourier space the solve is exact; any other mask (an instrument.PixelMask)
    goes to conjugate_gradient_wiener_filter.
    """
    if not isinstance(mask, DiagonalEB):
        return conjugate_gradient_wiener_filter(data, prior_covariance, noise_covariance,
                                                mask, beam, frequency_weights)
    data_precision = inverse_noise_power(noise_covariance, mask, beam, frequency_weights)
    weighted_data = inverse_noise_weighted_data(data, noise_covariance, mask, beam,
                                                frequency_weights)
    denominator = get_identity_like(prior_covariance) + prior_covariance * data_precision
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
    simulated_data = (apply_response(simulated_field, mask, beam, frequency_weights)
                      + simulated_noise)

    #x = x' + W * (d - d')
    correction = wiener_filter(data_residual - simulated_data, prior_covariance,
                               noise_covariance, mask, beam, frequency_weights)
    return simulated_field + correction

def least_squares_dust_field(data, noise_covariance, mask, beam, frequency_scaling):
    """(dust field, its covariance) of the joint least-squares fit of a CMB field and a dust
    field to the data, with no prior on either. The scattering sampler starts from a draw
    about this fit.

    Solves the 2 x 2 normal equations of (f, g) mode by mode. The covariance is a DiagonalEB.
    """
    cmb_frequency_weights = jnp.ones_like(frequency_scaling)
    cmb_precision = inverse_noise_power(noise_covariance, mask, beam, cmb_frequency_weights)
    dust_precision = inverse_noise_power(noise_covariance, mask, beam, frequency_scaling)
    #inverse_noise_power squares its weights, so sqrt(a_freq) gives the cross term sum_freq a_freq R N^-1 R
    cross_precision = inverse_noise_power(noise_covariance, mask, beam,
                                          jnp.sqrt(frequency_scaling))
    cmb_weighted_data = inverse_noise_weighted_data(data, noise_covariance, mask, beam,
                                                    cmb_frequency_weights)
    dust_weighted_data = inverse_noise_weighted_data(data, noise_covariance, mask, beam,
                                                     frequency_scaling)
    determinant = cmb_precision * dust_precision - cross_precision * cross_precision
    dust_estimate = pinv(determinant) * (cmb_precision * dust_weighted_data
                                         - cross_precision * cmb_weighted_data)
    return dust_estimate, pinv(determinant) * cmb_precision

def field_to_maps(field):
    """The real-space E and B maps of a single-channel Fourier field, shape (2, nside, nside)."""
    return jnp.stack([jfft.irfft2(field.polar_matrix_1), jfft.irfft2(field.polar_matrix_2)])

def maps_to_field(maps, field_like):
    """Inverse of field_to_maps."""
    return field_like.replace(polar_matrix_1 = jfft.rfft2(maps[0]),
                              polar_matrix_2 = jfft.rfft2(maps[1]))

@partial(jax.jit, static_argnames = ("gaussian_dust_prior",))
def hmc_sample_dust(dust_field, cmb_field, params, data, mask, beam, noise_covariance,
                    scalar_covariance, tensor_covariance, target_log_scattering_coefficients,
                    scattering_width, frequencies, dust_temperature, mass_matrix,
                    num_steps, step_size, rng_key, gaussian_dust_prior = False):
    """One Hamiltonian Monte Carlo update of the dust field on scattering_posterior, with
    the CMB field and the parameters fixed. Returns (dust field, whether it was accepted).

    The position is the pair of real-space dust maps, so the gradient is a plain jax.grad.
    `mass_matrix` is a DiagonalEB: the momentum is drawn from N(0, mass_matrix) and the
    velocity is mass_matrix^-1 * momentum. The mean of each map has no mass and never moves.
    """
    key_e, key_b, key_accept = jax.random.split(rng_key, 3)
    inverse_mass = pinv(mass_matrix)

    def logposterior(dust_maps):
        return scattering_posterior(data, cmb_field, maps_to_field(dust_maps, dust_field),
                                    mask, beam, noise_covariance = noise_covariance,
                                    scalar_covariance = scalar_covariance,
                                    tensor_covariance = tensor_covariance,
                                    target_log_scattering_coefficients = target_log_scattering_coefficients,
                                    scattering_width = scattering_width,
                                    frequencies = frequencies,
                                    dust_temperature = dust_temperature,
                                    gaussian_dust_prior = gaussian_dust_prior, **params)

    gradient = jax.grad(logposterior)

    def velocity(momentum):
        return field_to_maps(inverse_mass * maps_to_field(momentum, dust_field))

    def hamiltonian(dust_maps, momentum):
        return logposterior(dust_maps) - jnp.sum(momentum * velocity(momentum)) / 2

    #leapfrog integration: half a momentum kick, then alternate full position and momentum
    #steps; the last momentum kick is cut back to half a step afterwards
    def leapfrog_step(_, state):
        dust_maps, momentum = state
        dust_maps = dust_maps + step_size * velocity(momentum)
        momentum = momentum + step_size * gradient(dust_maps)
        return dust_maps, momentum

    start_maps = field_to_maps(dust_field)
    start_momentum = field_to_maps(draw_field(dust_field, mass_matrix, key_e, key_b))
    momentum = start_momentum + step_size * gradient(start_maps) / 2
    end_maps, momentum = jax.lax.fori_loop(0, num_steps, leapfrog_step, (start_maps, momentum))
    end_momentum = momentum - step_size * gradient(end_maps) / 2

    #a non-finite Hamiltonian makes the comparison False, so the step is rejected
    hamiltonian_change = hamiltonian(end_maps, end_momentum) - hamiltonian(start_maps, start_momentum)
    accepted = jnp.log(jax.random.uniform(key_accept)) < hamiltonian_change
    return maps_to_field(jnp.where(accepted, end_maps, start_maps), dust_field), accepted

##############################################################
#PARAMETER UPDATES
##############################################################

@jax.jit
def eval_gaussian_logposterior(params, cmb_field, dust_field, data, mask, beam, noise_covariance,
                      scalar_covariance, tensor_covariance, frequencies, dust_temperature):
    """gaussian_posterior at a {name: value} dict of the six parameters."""
    return gaussian_posterior(data, cmb_field, dust_field, mask, beam,
                              noise_covariance = noise_covariance,
                              scalar_covariance = scalar_covariance,
                              tensor_covariance = tensor_covariance,
                              frequencies = frequencies, dust_temperature = dust_temperature,
                              **params)

@partial(jax.jit, static_argnames = ("gaussian_dust_prior",))
def eval_scattering_logposterior(params, cmb_field, dust_field, data, mask, beam,
                                 noise_covariance, scalar_covariance, tensor_covariance,
                                 target_log_scattering_coefficients, scattering_width,
                                 frequencies, dust_temperature, gaussian_dust_prior = False):
    """scattering_posterior at a {name: value} dict of its parameters: r and
    frequency_exponent, plus the four dust parameters with `gaussian_dust_prior`."""
    return scattering_posterior(data, cmb_field, dust_field, mask, beam,
                                noise_covariance = noise_covariance,
                                scalar_covariance = scalar_covariance,
                                tensor_covariance = tensor_covariance,
                                target_log_scattering_coefficients = target_log_scattering_coefficients,
                                scattering_width = scattering_width,
                                frequencies = frequencies, dust_temperature = dust_temperature,
                                gaussian_dust_prior = gaussian_dust_prior, **params)

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
    samples drawn so far. `true_params` adds a line at the true value of each parameter it holds."""
    for name in names:
        figure = Figure(figsize = TRACE_FIGURE_SIZE)
        axis = figure.subplots()
        axis.plot(sample_histories[name], label = "chain")
        if true_params is not None and name in true_params:
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
        dust_data = apply_response(dust_field, mask, beam, frequency_scaling)
        cmb_field = gibbs_sample_field(data - dust_data, field_covariance, noise_covariance,
                                       mask, beam, cmb_frequency_weights, field_like, key_cmb)

        #2. draw the dust field given the CMB field
        cmb_data = apply_response(cmb_field, mask, beam, cmb_frequency_weights)
        dust_field = gibbs_sample_field(data - cmb_data, dust_covariance, noise_covariance,
                                        mask, beam, frequency_scaling, field_like, key_dust)

        #3. update the parameters one at a time at the new fields
        def logposterior_at(proposed_params):
            return eval_gaussian_logposterior(proposed_params, cmb_field, dust_field, data, mask, beam,
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

def sample_scattering_posterior(data, noise_covariance, scalar_covariance, tensor_covariance,
                                mask, beam, param_init, target_log_scattering_coefficients,
                                scattering_width = DEFAULT_SCATTERING_WIDTH,
                                proposal_widths = None,
                                param_bounds = PARAM_BOUNDS,
                                gaussian_dust_prior = False,
                                num_iterations = DEFAULT_NUM_ITERATIONS,
                                hmc_num_steps = DEFAULT_HMC_NUM_STEPS,
                                hmc_step_size = DEFAULT_HMC_STEP_SIZE,
                                seed = DEFAULT_CHAIN_SEED, frequencies = FREQUENCIES_GHZ,
                                dust_temperature = DUST_TEMPERATURE_K,
                                log_every = DEFAULT_LOG_EVERY, out_dir = None,
                                true_params = None):
    """Gibbs-sample the joint posterior P(f, g, r, frequency_exponent | d) of
    likelihoods.scattering_posterior, the model whose dust prior is an L2 loss on the log
    scattering coefficients of g.

    Each iteration draws the CMB field exactly (gibbs_sample_field), updates the dust field
    with one HMC step (hmc_sample_dust) and updates r and frequency_exponent with a 1D
    Metropolis step each. The dust field starts at least_squares_dust_field.

    Args:
        target_log_scattering_coefficients: The target of the dust term, as returned by
            likelihoods.log_scattering_coefficients.
        scattering_width: 1-sigma width of the L2 loss on each log coefficient.
        param_init: {name: value} starting point of r and frequency_exponent, and of the
            four dust parameters with `gaussian_dust_prior`.
        proposal_widths: As in sample_gaussian_posterior. None takes
            DEFAULT_SCATTERING_PROPOSAL_WIDTHS, or DEFAULT_PROPOSAL_WIDTHS with
            `gaussian_dust_prior`.
        gaussian_dust_prior: Keep the Gaussian dust prior next to the scattering loss (see
            scattering_posterior). The dust amplitudes and knees are then sampled too, and
            the prior's precision is added to the HMC mass matrix.
        hmc_num_steps: Leapfrog steps of the dust field's HMC update.
        hmc_step_size: Leapfrog step size, in units where the data's precision on the dust
            is the mass matrix.
        The other arguments are those of sample_gaussian_posterior.

    Returns:
        (sample_histories, acceptance_rates, logposterior_values), as
        sample_gaussian_posterior, for r and frequency_exponent (all six parameters with
        `gaussian_dust_prior`). acceptance_rates also
        holds the HMC acceptance rate of the dust field under "dust_field".
    """
    frequencies = jnp.asarray(frequencies, dtype = jnp.float64)
    cmb_frequency_weights = jnp.ones_like(frequencies)

    #a single-channel zero field: the template of every draw
    field_shape = data.polar_matrix_1.shape[-2:]
    zero_matrix = jnp.zeros(field_shape, dtype = data.polar_matrix_1.dtype)
    field_like = data.replace(polar_matrix_1 = zero_matrix, polar_matrix_2 = zero_matrix)

    param_order = PARAM_ORDER if gaussian_dust_prior else SCATTERING_PARAM_ORDER
    if proposal_widths is None:
        proposal_widths = (DEFAULT_PROPOSAL_WIDTHS if gaussian_dust_prior
                           else DEFAULT_SCATTERING_PROPOSAL_WIDTHS)
    params = {name: float(param_init[name]) for name in param_order}
    sample_histories = {name: [] for name in param_order}
    acceptance = {DUST_FIELD_NAME: [], **{name: [] for name in proposal_widths}}
    logposterior_values = []
    rng_key = jax.random.PRNGKey(seed)

    #the scattering coefficients of a zero field are zero, so the dust starts from the
    #least-squares fit to the data instead. A draw about the fit is used, not the fit
    #itself: the fit is the peak of the likelihood, where every mode has less energy than a
    #typical sample, and HMC trajectories that start there are systematically rejected
    rng_key, key_e, key_b = jax.random.split(rng_key, 3)
    frequency_scaling = dust_frequency_scaling(frequencies, params["frequency_exponent"],
                                               dust_temperature)
    dust_estimate, dust_estimate_covariance = least_squares_dust_field(
                 data, noise_covariance, mask, beam, frequency_scaling)
    dust_field = (field_like.replace(polar_matrix_1 = dust_estimate.polar_matrix_1,
                                     polar_matrix_2 = dust_estimate.polar_matrix_2)
                  + draw_field(field_like, dust_estimate_covariance, key_e, key_b))

    for iteration in range(1, num_iterations + 1):
        rng_key, key_cmb, key_dust = jax.random.split(rng_key, 3)

        #the CMB covariance and the dust frequency scaling at the current parameters
        field_covariance = cmb_field_covariance_from_r(params["r"], scalar_covariance,
                                                       tensor_covariance)
        frequency_scaling = dust_frequency_scaling(frequencies, params["frequency_exponent"],
                                                   dust_temperature)

        #1. draw the CMB field given the dust field
        dust_data = apply_response(dust_field, mask, beam, frequency_scaling)
        cmb_field = gibbs_sample_field(data - dust_data, field_covariance, noise_covariance,
                                       mask, beam, cmb_frequency_weights, field_like, key_cmb)

        #2. update the dust field given the CMB field with one HMC step, whose mass matrix
        #is the precision the data puts on the dust, plus that of the Gaussian prior if used
        mass_matrix = inverse_noise_power(noise_covariance, mask, beam, frequency_scaling)
        if gaussian_dust_prior:
            dust_covariance = dust_covariance_operator_like(params["dust_amplitude_ee"],
                                                            params["dust_knee_ee"],
                                                            params["dust_amplitude_bb"],
                                                            params["dust_knee_bb"],
                                                            scalar_covariance)
            mass_matrix = mass_matrix + pinv(dust_covariance)
        dust_field, dust_accepted = hmc_sample_dust(
            dust_field, cmb_field, params, data, mask, beam, noise_covariance,
            scalar_covariance, tensor_covariance, target_log_scattering_coefficients,
            scattering_width, frequencies, dust_temperature, mass_matrix,
            hmc_num_steps, hmc_step_size, key_dust,
            gaussian_dust_prior = gaussian_dust_prior)
        acceptance[DUST_FIELD_NAME].append(bool(dust_accepted))

        #3. update the parameters one at a time at the new fields
        def logposterior_at(proposed_params):
            return eval_scattering_logposterior(proposed_params, cmb_field, dust_field, data,
                                                mask, beam, noise_covariance, scalar_covariance,
                                                tensor_covariance,
                                                target_log_scattering_coefficients,
                                                scattering_width, frequencies, dust_temperature,
                                                gaussian_dust_prior = gaussian_dust_prior)

        logposterior = float(logposterior_at(params))
        for name, proposal_width in proposal_widths.items():
            rng_key, key_param = jax.random.split(rng_key)
            params, logposterior = metropolis_sample_parameter(logposterior_at, name, params,
                                                               logposterior, param_bounds[name],
                                                               proposal_width, acceptance,
                                                               key_param)

        for name in param_order:
            sample_histories[name].append(params[name])
        logposterior_values.append(logposterior)

        #4. diagnostics for hand-tuning the proposal widths and the HMC step size
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
        description = "Simulate multi-frequency E + B data with Gaussian or PySM dust and "
                      "Gibbs-sample r and the dust parameters from it with the Gaussian "
                      "dust model.")
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
    parser.add_argument("--dust_model", type = str, choices = DUST_MODELS,
                        default = GAUSSIAN_DUST,
                        help = "dust of the simulated data: a Gaussian field, or a PySM dust "
                               "map (non-Gaussian) on the BICEP/Keck field")
    parser.add_argument("--pysm_preset", type = str, default = DEFAULT_PYSM_PRESET,
                        help = "PySM dust model of --dust_model pysm")
    for name in PARAM_ORDER:
        parser.add_argument(f"--{name}", type = float, default = DEFAULT_TRUE_PARAMS[name],
                            help = f"true {name} of the simulated data (with PySM dust the "
                                   f"dust amplitudes and knees have no true value and this "
                                   f"only sets the chain's default starting point)")
        parser.add_argument(f"--{name}_init", type = float, default = None,
                            help = f"starting {name} of the chain (default: the true value)")
        parser.add_argument(f"--{name}_proposal_width", type = float,
                            default = DEFAULT_PROPOSAL_WIDTHS[name],
                            help = f"width of the Metropolis proposal of {name}")
    return parser.parse_args()

if __name__ == "__main__":

    arguments = vars(parse_arguments())
    os.makedirs(arguments["out_dir"], exist_ok = True)

    #the chain's starting point and its proposal widths
    param_init = {name: arguments[name] if arguments[f"{name}_init"] is None
                  else arguments[f"{name}_init"] for name in PARAM_ORDER}
    proposal_widths = {name: arguments[f"{name}_proposal_width"] for name in PARAM_ORDER}

    #the truth of the simulated data. PySM dust is not drawn from the power-law model, so
    #the dust amplitudes and knees are then not inputs of the simulation and have no truth
    true_names = PARAM_ORDER if arguments["dust_model"] == GAUSSIAN_DUST else PYSM_TRUE_PARAMS
    true_params = {name: arguments[name] for name in true_names}

    #generate a "ground truth" simulated data set
    sim = simulate_data(arguments["nside"], arguments["theta_pix"], arguments["map_seed"],
                        uk_arcmin = arguments["uk_arcmin"],
                        dust_temperature = arguments["dust_temperature"],
                        dust_model = arguments["dust_model"],
                        pysm_preset = arguments["pysm_preset"], **true_params)

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
        truth = f"{true_params[name]:.4g}" if name in true_params else "none"
        print(f"{name}: truth = {truth}, "
              f"mean = {np.mean(sample_histories[name]):.4g}, "
              f"std = {np.std(sample_histories[name]):.4g}, "
              f"acceptance rate = {acceptance_rates[name]:.2f}")
