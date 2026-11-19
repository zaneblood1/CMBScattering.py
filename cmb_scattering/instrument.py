"""The instrument response of the data model: masks and beams.

    d_freq = M_freq * B_freq * (f + a_freq * g) + n_freq

A beam B is a DiagonalEB (diagonal in Fourier space). A mask M is either a DiagonalEB, for
which the samplers solve their Wiener filters exactly mode by mode, or a PixelMask, which
multiplies the Q and U maps in real space and makes them use conjugate gradient. Both kinds
are applied to a field with `operator * field`.
"""

import flax
from flax import struct
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
jax.config.update("jax_enable_x64", True)
from cmb_lensing.constants import ARCMIN_PER_DEGREE
from cmb_lensing.matrix_operators import DiagonalEB
from cmb_lensing.simulate import beam_cls
from cmb_lensing.util import gen_ell_grid, get_fourier_weights, primal_eb2qu, primal_qu2eb
from cmb_scattering.simulate_dust import PYSM_TAPER_FRACTION, edge_taper

#FWHM of BICEP3's beam at 95 GHz: the median Gaussian width over the detectors is 0.161 deg,
#a FWHM of 0.379 deg (BICEP/Keck XV, https://arxiv.org/abs/2110.00482)
BICEP3_BEAM_FWHM_DEG = 0.379
BICEP3_BEAM_FWHM_ARCMIN = BICEP3_BEAM_FWHM_DEG * ARCMIN_PER_DEGREE
#fraction of the patch width over which apodized_border_mask ramps from zero to one (both
#edges together), the same shape as the taper of the PySM dust
DEFAULT_MASK_TAPER_FRACTION = PYSM_TAPER_FRACTION

@struct.dataclass
class PixelMask:
    """A mask that multiplies the Q and U maps pixel by pixel.

    `mask * field` takes an EB Fourier-space FlatS2 to Q and U maps, multiplies both by
    `pixel_map` and transforms back, so it mixes E and B. It also acts on a field with a
    leading frequency axis.

    Attributes:
        pixel_map: The mask in real space, an (nside, nside) array.
    """

    pixel_map: jnp.ndarray = flax.struct.field(default = None)

    #Actual telescopes in the real world measure polarization in the Q,U stokes parametrization
    #so the code applies the pixel masks there. Since E and B are then rotations of Q and U in fourier
    #space masking either Q or U affects both E and B
    def __mul__(self, field):
        nside, theta_pix = field.nside, field.theta_pix
        q_matrix, u_matrix = primal_eb2qu(field.polar_matrix_1, field.polar_matrix_2,
                                          nside, theta_pix)
        q_matrix = jfft.rfft2(self.pixel_map * jfft.irfft2(q_matrix))
        u_matrix = jfft.rfft2(self.pixel_map * jfft.irfft2(u_matrix))
        e_matrix, b_matrix = primal_qu2eb(q_matrix, u_matrix, nside, theta_pix)
        return field.replace(polar_matrix_1 = e_matrix, polar_matrix_2 = b_matrix)

def apodized_border_mask(nside, taper_fraction = DEFAULT_MASK_TAPER_FRACTION):
    """A PixelMask that is one in the middle of the patch and falls to zero at its edges
    along a cosine ramp (a Tukey window in each direction)."""
    return PixelMask(pixel_map = jnp.asarray(edge_taper(nside, taper_fraction)))

def gaussian_beam(nside, theta_pix, fwhm_arcmin = BICEP3_BEAM_FWHM_ARCMIN):
    """A Gaussian beam of the given FWHM as a DiagonalEB, the same for E and B.

    For a different beam in every channel, stack the blocks of several beams along a
    leading frequency axis.
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #beam_cls is the beam's power transfer function B_ell^2
    beam_matrix = jnp.sqrt(beam_cls(fwhm_arcmin, ell_grid))
    return DiagonalEB(fourier_weights = get_fourier_weights(ell_grid.shape), nside = nside,
                      theta_pix = theta_pix, pix_width = pix_width,
                      matrix_EE = beam_matrix, matrix_BB = beam_matrix)
