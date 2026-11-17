import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
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
