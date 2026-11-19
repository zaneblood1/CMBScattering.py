from functools import lru_cache
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
jax.config.update("jax_enable_x64", True)
#kymatio's top-level `kymatio.jax` also imports its 3D transform, which fails on current
#scipy (scipy.special.sph_harm was removed), so the 2D frontend is imported directly
from kymatio.scattering2d.frontend.jax_frontend import ScatteringJax2D
from cmb_lensing.fields import dot
from cmb_lensing.matrix_operators import log_det, pinv
from cmb_scattering.simulate_dust import (DUST_TEMPERATURE_K, FREQUENCIES_GHZ,
                                          cmb_field_covariance_from_r,
                                          dust_covariance_operator_like, dust_frequency_scaling,
                                          data_model)

#Gaussian prior on the dust frequency exponent beta_d: the BK18 constraint is
#1.49 +0.13 / -0.12 (BICEP/Keck XIII, https://arxiv.org/abs/2110.00483) and the width is
#the larger of the two errors
FREQUENCY_EXPONENT_PRIOR_MEAN = 1.49
FREQUENCY_EXPONENT_PRIOR_WIDTH = 0.13

#the wavelet scattering transform of the dust maps (kymatio's Scattering2D, Morlet
#wavelets, orders 0 to 2): the wavelets span 2^1 to 2^5 pixels in 4 orientations, which
#gives 180 first- and second-order coefficients per map
SCATTERING_NUM_SCALES = 5
SCATTERING_NUM_ORIENTATIONS = 4
#1-sigma width of the L2 loss on each log scattering coefficient. Hand-chosen, not derived
DEFAULT_SCATTERING_WIDTH = 0.1

#gaussian log-posterior (log-likelihood + log-prior) for polarization-only (E + B) multi-frequency data
def gaussian_posterior(data_field, cmb_field, dust_field, mask, beam,
                        dust_amplitude_ee, dust_knee_ee, dust_amplitude_bb, dust_knee_bb,
                        frequency_exponent, r,
                        noise_covariance, scalar_covariance, tensor_covariance,
                        frequencies = FREQUENCIES_GHZ, dust_temperature = DUST_TEMPERATURE_K):
    """Joint log-density of the data, the CMB field f and the dust field g, including the
    Gaussian prior on the frequency exponent.

    -1/2 * [sum_over_freq {(d_freq - M * B * (f + a_freq * g))^2 / C_n_freq}
            + f^2 / C_f(r) + g^2 / C_g(A_ee, alpha_ee, A_bb, alpha_bb)
            + sum_over_freq {logdet(C_n_freq)} + logdet(C_f(r)) 
            + logdet(C_g(A_ee, alpha_ee, A_bb, alpha_bb))]
            + frequency_exponent_logprior(frequency_exponent)

    C_g has its own D_ell power law for EE (dust_amplitude_ee, dust_knee_ee) and for BB
    (dust_amplitude_bb, dust_knee_bb). `data_field` and `noise_covariance` carry a leading frequency axis; the sums over
    channels happen inside `dot` and `log_det`. `scalar_covariance` and `tensor_covariance`
    are single-channel DiagonalEB operators with C_f(r) = scalar + r * tensor.
    """
    field_covariance = cmb_field_covariance_from_r(r, scalar_covariance, tensor_covariance)
    dust_covariance = dust_covariance_operator_like(dust_amplitude_ee, dust_knee_ee,
                                                    dust_amplitude_bb, dust_knee_bb,
                                                    scalar_covariance)
    frequency_scaling = dust_frequency_scaling(frequencies, frequency_exponent, dust_temperature)

    #the three log-determinants (the noise one sums over the channels)
    noise_logdet = log_det(noise_covariance)
    field_logdet = log_det(field_covariance)
    dust_logdet = log_det(dust_covariance)

    #the f^2 and g^2 contributions
    field_product = dot(cmb_field, pinv(field_covariance) * cmb_field)
    dust_product = dot(dust_field, pinv(dust_covariance) * dust_field)

    #the data term, summed over the channels
    difference = data_field - data_model(cmb_field, dust_field, frequency_scaling, mask, beam)
    data_product = dot(difference, pinv(noise_covariance) * difference)

    value = -jnp.real(data_product + field_product + dust_product
                      + noise_logdet + field_logdet + dust_logdet) / 2
    #the Gaussian prior on the frequency exponent
    return value + frequency_exponent_logprior(frequency_exponent)

#Gaussian prior on the frequency exponent with a specified width from experimental results
def frequency_exponent_logprior(frequency_exponent, mean = FREQUENCY_EXPONENT_PRIOR_MEAN,
                                width = FREQUENCY_EXPONENT_PRIOR_WIDTH):
    #`width` is the 1-sigma uncertainty; the normalization does not depend on the exponent
    return -(frequency_exponent - mean)**2 / (2 * width**2)

##############################################################
#SCATTERING TRANSFORM POSTERIOR
##############################################################

@lru_cache(maxsize = None)
def scattering_transform(nside):
    """kymatio's 2D scattering transform for nside x nside maps (built once per nside)."""
    return ScatteringJax2D(J = SCATTERING_NUM_SCALES, shape = (nside, nside),
                           L = SCATTERING_NUM_ORIENTATIONS)

def log_scattering_coefficients(dust_field):
    """Log of the first- and second-order scattering coefficients of the E and the B map of
    a dust field, averaged over the patch: an array of shape (2, number of coefficients).

    The zeroth-order coefficient is the local mean of the map, which is not positive, and
    is left out. E and B are transformed separately; no coefficient couples them.
    """
    dust_maps = jnp.stack([jfft.irfft2(dust_field.polar_matrix_1),
                           jfft.irfft2(dust_field.polar_matrix_2)])
    coefficients = scattering_transform(dust_maps.shape[-1])(dust_maps)
    #This scattering transform package gives an output based on position. The first axis
    #denotes which map we are on (E v.s. B). The second axis represents the 181 scattering coefficients
    #for this combination of L and J. The 3rd and 4th axes index different positions on the map (for every
    #32 x 32 pixel region for 16 total sub-divisions for 128 x 128 pixel maps). We do not want to encode
    #position in our learned maps, so we average over the positions axes (2, 3) or equivalently (-2, -1).
    #We then keep all the rows (E v.s. B maps) but discard the zeroth order scattering coefficient since it is
    #the smoothed map itself and can cause problems with negative logs etc.
    return jnp.log(jnp.mean(coefficients, axis = (-2, -1))[:, 1:])

#non-Gaussian log-posterior: the dust prior is an L2 loss on its log scattering coefficients
def scattering_posterior(data_field, cmb_field, dust_field, mask, beam,
                         frequency_exponent, r,
                         noise_covariance, scalar_covariance, tensor_covariance,
                         target_log_scattering_coefficients,
                         scattering_width = DEFAULT_SCATTERING_WIDTH,
                         frequencies = FREQUENCIES_GHZ, dust_temperature = DUST_TEMPERATURE_K,
                         gaussian_dust_prior = False,
                         dust_amplitude_ee = None, dust_knee_ee = None,
                         dust_amplitude_bb = None, dust_knee_bb = None):
    """Joint log-density of the data, the CMB field f and the dust field g, in which the
    Gaussian dust prior of gaussian_posterior is replaced by an L2 loss between the log
    scattering coefficients of g and a target set.

    -1/2 * [sum_over_freq {(d_freq - M * B * (f + a_freq * g))^2 / C_n_freq}
            + f^2 / C_f(r)
            + sum_over_freq {logdet(C_n_freq)} + logdet(C_f(r))
            + sum_over_coefficients {(log S(g) - log S_target)^2} / scattering_width^2]
            + frequency_exponent_logprior(frequency_exponent)

    The scattering term has no free parameters (no amplitudes or knees), so its
    normalization is a constant and is left out. `target_log_scattering_coefficients` has
    the shape log_scattering_coefficients returns. The other arguments are those of
    gaussian_posterior.

    `gaussian_dust_prior = True` keeps the Gaussian dust prior of gaussian_posterior as well,
    adding g^2 / C_g + logdet(C_g) inside the brackets, so the scattering loss multiplies a
    normalized density of g. The four dust amplitude / knee arguments are needed only then.
    The integral of that product over g still depends on the dust parameters and is not
    divided out.
    """
    field_covariance = cmb_field_covariance_from_r(r, scalar_covariance, tensor_covariance)
    frequency_scaling = dust_frequency_scaling(frequencies, frequency_exponent, dust_temperature)

    #the two log-determinants (the noise one sums over the channels)
    noise_logdet = log_det(noise_covariance)
    field_logdet = log_det(field_covariance)

    #the f^2 contribution
    field_product = dot(cmb_field, pinv(field_covariance) * cmb_field)

    #the data term, summed over the channels
    difference = data_field - data_model(cmb_field, dust_field, frequency_scaling, mask, beam)
    data_product = dot(difference, pinv(noise_covariance) * difference)

    #the dust term: L2 loss of the log scattering coefficients
    scattering_difference = log_scattering_coefficients(dust_field) - target_log_scattering_coefficients
    scattering_loss = jnp.sum(scattering_difference**2) / scattering_width**2

    #the optional Gaussian dust prior, with its log-determinant
    gaussian_dust_term = 0.0
    if gaussian_dust_prior:
        dust_covariance = dust_covariance_operator_like(dust_amplitude_ee, dust_knee_ee,
                                                        dust_amplitude_bb, dust_knee_bb,
                                                        scalar_covariance)
        gaussian_dust_term = jnp.real(dot(dust_field, pinv(dust_covariance) * dust_field)
                                      + log_det(dust_covariance))

    value = -(jnp.real(data_product + field_product + noise_logdet + field_logdet)
              + scattering_loss + gaussian_dust_term) / 2
    #the Gaussian prior on the frequency exponent
    return value + frequency_exponent_logprior(frequency_exponent)
