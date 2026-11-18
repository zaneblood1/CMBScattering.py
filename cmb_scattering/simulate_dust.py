"""Multi-frequency, polarization-only (E + B) data model with a Gaussian dust foreground.

    d_freq = M_freq * B_freq * (f + a_freq * g) + n_freq

    f ~ N(0, C_f(r))      the CMB, identical in every channel (maps are in uK_CMB)
    g ~ N(0, C_g)         the dust at the reference frequency (353 GHz)
    n_freq ~ N(0, C_n)    the noise, independent between channels

Fields are cmb_lensing FlatS2 objects (EB, Fourier space) and operators are DiagonalEB.
Anything with one entry per channel (the data, the noise covariance and, in the future, the
mask and the beam) carries a leading frequency axis, (n_freq, nside, nside//2+1). The field
and operator arithmetic of cmb_lensing is element-wise, so it broadcasts over that axis.
"""

from functools import lru_cache
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
jax.config.update("jax_enable_x64", True)
import scipy.constants
from scipy.signal.windows import tukey
from cmb_lensing.constants import (ARCMIN_PER_DEGREE, CAMB_LMAX, DEFAULT_A_LENSE,
                                   DEFAULT_K_PIVOT, DEFAULT_MAX_ELL, DEFAULT_MNU,
                                   DEFAULT_TAUREIO, GROUND_TRUTH, Basis, Parametrization)
from cmb_lensing.fields import FlatS2
from cmb_lensing.matrix_operators import DiagonalEB, get_identity_like
from cmb_lensing.simulate import (_CAMB_COLS, _run_camb, covar_matrix_from_cls, dl2cl,
                                  field_from_covar_single_key)
from cmb_lensing.util import gen_ell_grid, get_fourier_weights

#Sources of the constants below:
#    BK18        BICEP/Keck XIII, https://arxiv.org/abs/2110.00483
#    Planck XI   Planck 2018 results XI, polarized dust foregrounds, https://arxiv.org/abs/1801.04945
#    Fixsen      Fixsen 2009, the temperature of the CMB, https://arxiv.org/abs/0911.1955

#the frequency channels of the data, in GHz (the BICEP/Keck bands of BK18)
FREQUENCIES_GHZ = (95.0, 150.0, 220.0)
#the frequency the dust field g and its amplitude are defined at, in GHz (BK18's A_d,353)
DUST_REFERENCE_FREQUENCY_GHZ = 353.0
#temperature of the dust modified blackbody, in K (BK18 fixes T_d = 19.6 K)
DUST_TEMPERATURE_K = 19.6
#temperature of the CMB monopole, in K (sets the brightness to uK_CMB conversion). Fixsen
#measures 2.72548 +/- 0.00057 K
CMB_TEMPERATURE_K = 2.7255
#the multipole the dust amplitudes are quoted at (BK18 and Planck XI both use ell = 80)
DUST_PIVOT_ELL = 80
#measured dust power laws, for reference when choosing simulation inputs. Planck XI (Eq. 1
#and Table 1, largest sky region LR71) fits D_ell = A * (ell / 80)^(alpha + 2) with
#alpha_EE = -2.42 +/- 0.02 and alpha_BB = -2.54 +/- 0.02, i.e. the D_ell knees below, and a
#BB to EE amplitude ratio of 0.53 +/- 0.01
PLANCK_DUST_KNEE_EE = -0.42
PLANCK_DUST_KNEE_BB = -0.54
PLANCK_DUST_BB_TO_EE_AMPLITUDE_RATIO = 0.53
#the tensor covariance is the CAMB tensor spectrum at this r, so C_f(r) = C_scalar + r * C_tensor
TENSOR_TEMPLATE_R = 1.0
GHZ_TO_HZ = 1e9
#default polarization white noise level, in uK-arcmin
DEFAULT_UK_ARCMIN = 3.0
#the E and B components of every covariance, in that order
POLARIZATION_COMPONENTS = ("EE", "BB")

#the dust of the simulated data: a Gaussian random field with a power-law spectrum, or a
#PySM dust map (real-sky, non-Gaussian) projected onto the flat patch
GAUSSIAN_DUST = "gaussian"
PYSM_DUST = "pysm"
DUST_MODELS = (GAUSSIAN_DUST, PYSM_DUST)
#the PySM dust model: d9 is the Planck GNILC based model with one spectral index and one
#temperature over the whole sky (https://pysm3.readthedocs.io/en/latest/models/)
DEFAULT_PYSM_PRESET = "d9"
#centre of the PySM patch: the BICEP/Keck field, RA 0h, Dec -57.5 deg (BK18,
#https://arxiv.org/abs/2110.00483)
PYSM_PATCH_CENTER_RA_DEG = 0.0
PYSM_PATCH_CENTER_DEC_DEG = -57.5
#the healpix maps the patch is read from have at least this many pixels per flat pixel width
HEALPIX_PIXELS_PER_FLAT_PIXEL = 2
#fraction of the patch width over which the PySM dust is tapered to zero (both edges together)
PYSM_TAPER_FRACTION = 0.2
#the Galactic plane is removed before the E / B split on the sphere: the PySM maps are
#multiplied by zero below the first Galactic latitude (degrees), by one above the second,
#and by a cosine ramp in between. The patch must lie above the second
GALACTIC_MASK_ZERO_LATITUDE_DEG = 15.0
GALACTIC_MASK_FULL_LATITUDE_DEG = 30.0
DEGREES_PER_RADIAN = 180 / np.pi

##############################################################
#COVARIANCES
##############################################################

def cmb_field_covariance_from_r(r, scalar_covariance, tensor_covariance):
    #the operator goes on the left so its own __mul__ handles a float or a jax scalar r
    return scalar_covariance + tensor_covariance * r

def gaussian_dust_covariance(dust_amplitude, dust_knee, ells, ell_grid, pix_width):
    """The dust covariance matrix of one component (EE or BB) on the rfft2 grid.

    The power law is in D_ell, the BK18 convention:
    D_ell^dd = ell (ell + 1) cl^dd / (2 pi) = dust_amplitude * (ell / 80)^dust_knee at the
    reference frequency. It is converted to cl^dd and interpolated onto `ell_grid` in the
    units of every other covariance (covar_matrix_from_cls).
    """
    nside = ell_grid.shape[0]
    dust_dls = dust_amplitude * (ells / DUST_PIVOT_ELL)**dust_knee
    dust_cls = dust_dls * 2 * jnp.pi / (ells * (ells + 1))
    dust_covariance = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, dust_cls, origin_value = 0)
    return dust_covariance

def dust_covariance_operator_like(dust_amplitude_ee, dust_knee_ee, dust_amplitude_bb,
                                  dust_knee_bb, operator_like):
    """C_g as a DiagonalEB on the grid of `operator_like` (any single-channel DiagonalEB),
    with its own power law for EE and for BB."""
    ell_grid, pix_width = gen_ell_grid(operator_like.nside, operator_like.theta_pix)
    ells = jnp.arange(2, DEFAULT_MAX_ELL).astype(jnp.float64)
    dust_covariance_ee = gaussian_dust_covariance(dust_amplitude_ee, dust_knee_ee, ells,
                                                  ell_grid, pix_width)
    dust_covariance_bb = gaussian_dust_covariance(dust_amplitude_bb, dust_knee_bb, ells,
                                                  ell_grid, pix_width)
    return operator_like.replace(matrix_EE = dust_covariance_ee, matrix_BB = dust_covariance_bb)

def white_noise_covariance(nside, theta_pix, uk_arcmin = DEFAULT_UK_ARCMIN,
                           num_frequencies = len(FREQUENCIES_GHZ)):
    """White noise of the same amplitude for E and B in every channel.

    `uk_arcmin` is the polarization noise level itself. The blocks have one entry per channel,
    (num_frequencies, nside, nside//2+1), so a different spectrum per channel and per E / B
    only needs different blocks. Noise is uncorrelated between channels.
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    noise_level = jnp.deg2rad(uk_arcmin / ARCMIN_PER_DEGREE)**2 / pix_width**2
    #no noise in the mean of the map, like every other covariance
    noise_matrix = jnp.full(ell_grid.shape, noise_level).at[0, 0].set(0.0)
    noise_matrix = jnp.broadcast_to(noise_matrix, (num_frequencies,) + ell_grid.shape)
    return DiagonalEB(fourier_weights = get_fourier_weights(ell_grid.shape), nside = nside,
                      theta_pix = theta_pix, pix_width = pix_width,
                      matrix_EE = noise_matrix, matrix_BB = noise_matrix)

def _covariance_from_cls(nside, pix_width, ell_grid, ells, cls):
    #covar_matrix_from_cls interpolates in log(Cl), so it is only given the multipoles where
    #the spectrum is positive (CAMB's tensor spectra stop at max_l_tensor) and the matrix is
    #zero past the last of them. Eager only: the boolean indexing cannot be traced
    positive = cls > 0
    if not bool(jnp.any(positive)):
        return jnp.zeros(ell_grid.shape)
    positive_ells = ells[positive]
    covariance = covar_matrix_from_cls(nside, pix_width, ell_grid, positive_ells,
                                       cls[positive], origin_value = 0)
    return jnp.where(ell_grid <= positive_ells[-1], covariance, 0.0)

def cmb_covariances(nside, theta_pix, params = GROUND_TRUTH):
    """(scalar_covariance, tensor_covariance) as DiagonalEB operators, from one CAMB run.

    The scalar covariance is the LENSED scalar spectrum (lensing treated as Gaussian) and the
    tensor covariance is the tensor spectrum at r = 1, so that
    C_f(r) = scalar_covariance + r * tensor_covariance. `params` holds the five LCDM
    parameters in cmb_lensing's names (PARAM_ORDER).
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    power_spectra, _ = _run_camb(None, params["ombh2"], params["omch2"],
                                 params["theta_MC_100"] / 100, TENSOR_TEMPLATE_R, DEFAULT_MNU,
                                 DEFAULT_TAUREIO, float(jnp.exp(params["logA"])) * 1e-10, 0,
                                 params["ns"], CAMB_LMAX, DEFAULT_K_PIVOT, DEFAULT_A_LENSE)
    ells = jnp.arange(2, CAMB_LMAX).astype(jnp.float64)

    def covariance_matrices(spectra_name):
        matrices = {}
        for component in POLARIZATION_COMPONENTS:
            dls = jnp.asarray(power_spectra[spectra_name])[:, _CAMB_COLS[component]]
            cls = dl2cl(dls, CAMB_LMAX, CAMB_LMAX)
            matrices[component] = _covariance_from_cls(nside, pix_width, ell_grid, ells, cls)
        return DiagonalEB(fourier_weights = get_fourier_weights(ell_grid.shape), nside = nside,
                          theta_pix = theta_pix, pix_width = pix_width,
                          matrix_EE = matrices["EE"], matrix_BB = matrices["BB"])

    scalar_covariance = covariance_matrices("lensed_scalar")
    tensor_covariance = covariance_matrices("tensor")
    return scalar_covariance, tensor_covariance

##############################################################
#DUST FREQUENCY DEPENDENCE
##############################################################

def planck_function(frequency_ghz, temperature):
    """Blackbody spectral radiance B_freq(T), in SI units."""
    frequency = frequency_ghz * GHZ_TO_HZ
    x = scipy.constants.h * frequency / (scipy.constants.k * temperature)
    return 2 * scipy.constants.h * frequency**3 / scipy.constants.c**2 / jnp.expm1(x)

def planck_function_temperature_derivative(frequency_ghz, temperature):
    """dB_freq/dT, the factor that converts a thermodynamic temperature to a brightness."""
    frequency = frequency_ghz * GHZ_TO_HZ
    x = scipy.constants.h * frequency / (scipy.constants.k * temperature)
    return planck_function(frequency_ghz, temperature) * x * jnp.exp(x) / (temperature * jnp.expm1(x))

def dust_frequency_scaling(frequencies, frequency_exponent,
                           dust_temperature = DUST_TEMPERATURE_K,
                           reference_frequency = DUST_REFERENCE_FREQUENCY_GHZ):
    """a_freq of each channel, with g_freq = a_freq * g and a = 1 at the reference frequency.

    a_freq = (freq / freq_0)^beta_d * B_freq(T_d) / B_freq0(T_d), a modified blackbody,
    times [dB/dT at freq_0] / [dB/dT at freq] at the CMB temperature, which puts every
    channel in uK_CMB (the units in which the CMB is the same in every channel).
    """
    frequencies = jnp.asarray(frequencies, dtype = jnp.float64)
    power_law = (frequencies / reference_frequency)**frequency_exponent
    blackbody_ratio = (planck_function(frequencies, dust_temperature)
                       / planck_function(reference_frequency, dust_temperature))
    to_cmb_units = (planck_function_temperature_derivative(reference_frequency, CMB_TEMPERATURE_K)
                    / planck_function_temperature_derivative(frequencies, CMB_TEMPERATURE_K))
    return power_law * blackbody_ratio * to_cmb_units

def scale_dust_to_frequencies(dust_field, frequency_scaling):
    """g_freq = a_freq * g: the dust in every channel, with a leading frequency axis."""
    scaling = frequency_scaling[:, None, None]
    return dust_field.replace(polar_matrix_1 = scaling * dust_field.polar_matrix_1,
                              polar_matrix_2 = scaling * dust_field.polar_matrix_2)

def data_model(cmb_field, dust_field, frequency_scaling, mask, beam):
    """The noise-free data of every channel, M * B * (f + a_freq * g)."""
    dust_by_frequency = scale_dust_to_frequencies(dust_field, frequency_scaling)
    #the dust goes on the left so the sum takes its (n_freq, nside, nside//2+1) shape
    return mask * beam * (dust_by_frequency + cmb_field)

##############################################################
#PYSM DUST
##############################################################

def galactic_plane_mask(healpix_nside):
    """A healpix map that is zero in the Galactic plane, one at high Galactic latitude and a
    cosine ramp in between (the GALACTIC_MASK_* latitudes)."""
    import healpy

    colatitude, _ = healpy.pix2ang(healpix_nside, np.arange(healpy.nside2npix(healpix_nside)))
    latitude_deg = np.abs(90 - colatitude * DEGREES_PER_RADIAN)
    ramp = np.clip((latitude_deg - GALACTIC_MASK_ZERO_LATITUDE_DEG)
                   / (GALACTIC_MASK_FULL_LATITUDE_DEG - GALACTIC_MASK_ZERO_LATITUDE_DEG), 0, 1)
    return (1 - np.cos(np.pi * ramp)) / 2

@lru_cache(maxsize = None)
def pysm_dust_eb_maps(preset, healpix_nside, lmin, lmax):
    """(E map, B map): the PySM dust polarization at 353 GHz as two scalar healpix maps in
    uK_CMB and Galactic coordinates, holding only the multipoles lmin <= ell <= lmax.

    The E / B split is done on the sphere, so no patch edge can leak E into B. Scalar E and
    B maps are non-local, though: the Galactic plane is a thousand times brighter than the
    high-latitude sky and would dominate the E map far from it, so it is masked first
    (galactic_plane_mask). Cached, because the PySM templates do not depend on the seed. The
    first call downloads the templates into the astropy cache.
    """
    import healpy
    import pysm3
    import pysm3.units as units

    sky = pysm3.Sky(nside = healpix_nside, preset_strings = [preset], output_unit = units.uK_CMB)
    stokes_maps = sky.get_emission(DUST_REFERENCE_FREQUENCY_GHZ * units.GHz).value
    _, alm_e, alm_b = healpy.map2alm(stokes_maps * galactic_plane_mask(healpix_nside), lmax = lmax)
    keep_multipoles = (np.arange(lmax + 1) >= lmin).astype(float)
    e_map = healpy.alm2map(healpy.almxfl(alm_e, keep_multipoles), healpix_nside, lmax = lmax, pol = False)
    b_map = healpy.alm2map(healpy.almxfl(alm_b, keep_multipoles), healpix_nside, lmax = lmax, pol = False)
    return e_map, b_map

def flat_patch_sky_coordinates(nside, theta_pix, center_ra_deg = PYSM_PATCH_CENTER_RA_DEG,
                               center_dec_deg = PYSM_PATCH_CENTER_DEC_DEG):
    """Galactic (longitude, latitude) in degrees of every pixel of the flat patch.

    The patch is laid out as a longitude / latitude grid about the equator of a frame whose
    origin is the patch centre, which is only undistorted near that centre.
    """
    import healpy
    from astropy.coordinates import SkyCoord

    center = SkyCoord(ra = center_ra_deg, dec = center_dec_deg, unit = "deg").galactic
    offsets_deg = (np.arange(nside) - nside / 2) * theta_pix / ARCMIN_PER_DEGREE
    local_longitude, local_latitude = np.meshgrid(offsets_deg, offsets_deg)
    #the rotator takes the patch centre to (0, 0), so its inverse takes local to Galactic
    to_local_frame = healpy.Rotator(rot = [center.l.deg, center.b.deg], deg = True)
    longitude, latitude = to_local_frame.I(local_longitude.ravel(), local_latitude.ravel(),
                                           lonlat = True)
    return longitude.reshape(nside, nside), latitude.reshape(nside, nside)

def edge_taper(nside, taper_fraction = PYSM_TAPER_FRACTION):
    """An nside x nside window that is one in the middle of the patch and falls smoothly to
    zero at its edges, which makes a cut-out of the sky periodic on the FFT grid."""
    window = tukey(nside, alpha = taper_fraction)
    return np.outer(window, window)

def pysm_dust_field(field_like, preset = DEFAULT_PYSM_PRESET,
                    taper_fraction = PYSM_TAPER_FRACTION):
    """The PySM dust at 353 GHz on the flat patch of `field_like`, as a FlatS2 in Fourier
    space (uK_CMB).

    The E and B healpix maps are read at the patch's pixels, their mean over the patch is
    removed and they are tapered to zero at its edges. Only the multipoles the flat grid can
    hold are kept, from the patch's fundamental to its Nyquist multipole. The dust is the
    real sky's, so it is the same on every call. healpy's sign convention
    for B need not match cmb_lensing's; nothing at the power-spectrum level depends on it.
    """
    import healpy

    #We need to achieve a higher resolution on the curved sky than our target resolution on the flat
    #sky (called "oversampling") in order to avoid projection effects / numerical artifacts
    nside, theta_pix = field_like.nside, field_like.theta_pix
    healpix_nside = 1
    while healpy.nside2resol(healpix_nside, arcmin = True) > theta_pix / HEALPIX_PIXELS_PER_FLAT_PIXEL:
        healpix_nside *= 2
    #the fundamental and the Nyquist multipole of the flat grid: nothing outside them can be
    #represented there
    lmin = int(2 * np.pi / (nside * field_like.pix_width))
    lmax = int(np.pi / field_like.pix_width)
    e_healpix, b_healpix = pysm_dust_eb_maps(preset, healpix_nside, lmin, lmax)

    longitude, latitude = flat_patch_sky_coordinates(nside, theta_pix)
    taper = edge_taper(nside, taper_fraction)

    def flat_fourier_matrix(healpix_map):
        flat_map = healpy.get_interp_val(healpix_map, longitude, latitude, lonlat = True)
        #remove the patch's mean before tapering, or the taper would turn it into a
        #patch-sized plateau of spurious large-scale power
        flat_map = flat_map - np.sum(flat_map * taper) / np.sum(taper)
        #no power in the mean of the map, like every field the model draws
        return jfft.rfft2(jnp.asarray(flat_map * taper)).at[0, 0].set(0.0)

    return field_like.replace(polar_matrix_1 = flat_fourier_matrix(e_healpix),
                              polar_matrix_2 = flat_fourier_matrix(b_healpix))

##############################################################
#SIMULATION
##############################################################

def draw_field(field_like, covariance, key_e, key_b):
    #one Gaussian E/B draw from a single-channel DiagonalEB covariance, in Fourier space
    nside = field_like.nside
    e_matrix = jfft.rfft2(field_from_covar_single_key(nside, covariance.matrix_EE, key_e))
    b_matrix = jfft.rfft2(field_from_covar_single_key(nside, covariance.matrix_BB, key_b))
    return field_like.replace(polar_matrix_1 = e_matrix, polar_matrix_2 = b_matrix)

def draw_noise(field_like, noise_covariance, rng_key):
    #an independent noise draw per channel and per component
    nside = field_like.nside
    num_frequencies = noise_covariance.matrix_EE.shape[0]
    keys_e, keys_b = jax.random.split(rng_key, (2, num_frequencies))

    def draw(key, covariance_matrix):
        return jfft.rfft2(field_from_covar_single_key(nside, covariance_matrix, key))

    return field_like.replace(polar_matrix_1 = jax.vmap(draw)(keys_e, noise_covariance.matrix_EE),
                              polar_matrix_2 = jax.vmap(draw)(keys_b, noise_covariance.matrix_BB))

def simulate_data(nside, theta_pix, master_seed, r, frequency_exponent,
                  dust_amplitude_ee = None, dust_knee_ee = None,
                  dust_amplitude_bb = None, dust_knee_bb = None,
                  uk_arcmin = DEFAULT_UK_ARCMIN,
                  frequencies = FREQUENCIES_GHZ, dust_temperature = DUST_TEMPERATURE_K,
                  params = GROUND_TRUTH, dust_model = GAUSSIAN_DUST,
                  pysm_preset = DEFAULT_PYSM_PRESET):
    """Simulate multi-frequency E + B data on a flat nside x nside patch.

    Args:
        nside: Number of pixels on a side.
        theta_pix: Pixel width in arcmin.
        master_seed: Root seed of every random draw.
        r: Tensor-to-scalar ratio.
        frequency_exponent: beta_d of the dust modified blackbody, which scales the dust
            between the channels in both dust models.
        dust_amplitude_ee: Dust EE D_l = l (l + 1) C_l / (2 pi) at ell = 80 and 353 GHz,
            in uK_CMB^2. Gaussian dust only, like the three parameters below.
        dust_knee_ee: Power-law index of the dust EE D_l.
        dust_amplitude_bb: The same for BB (BK18's A_d).
        dust_knee_bb: Power-law index of the dust BB D_l (BK18's alpha_d).
        uk_arcmin: Polarization white noise level of every channel, in uK-arcmin.
        frequencies: The channels, in GHz.
        dust_temperature: T_d of the dust modified blackbody, in K.
        params: The LCDM parameters of the CMB spectra.
        dust_model: "gaussian" draws the dust as a Gaussian field with the power-law
            spectra above. "pysm" takes the 353 GHz dust of a PySM model on the BICEP/Keck
            field (pysm_dust_field): non-Gaussian, and the same on every call, so the seed
            then only changes the CMB and the noise.
        pysm_preset: The PySM dust model of dust_model = "pysm".

    Returns:
        A dict with the fields `data` (leading frequency axis), `cmb_field` and `dust_field`,
        the operators `noise_covariance`, `scalar_covariance`, `tensor_covariance`,
        `field_covariance`, `dust_covariance` (None for PySM dust), `mask` and `beam`,
        `frequencies` / `frequency_scaling`, and `dust_model`. All fields are in Fourier
        space.
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    frequencies = jnp.asarray(frequencies, dtype = jnp.float64)

    scalar_covariance, tensor_covariance = cmb_covariances(nside, theta_pix, params)
    field_covariance = cmb_field_covariance_from_r(r, scalar_covariance, tensor_covariance)
    noise_covariance = white_noise_covariance(nside, theta_pix, uk_arcmin, frequencies.shape[0])
    #identity for now: a per-channel mask or beam is a block with a leading frequency axis
    mask = get_identity_like(scalar_covariance)
    beam = get_identity_like(scalar_covariance)

    field_like = FlatS2(fourier_weights = get_fourier_weights(ell_grid.shape), nside = nside,
                        theta_pix = theta_pix, pix_width = pix_width, basis = Basis.FOURIER,
                        parametrization = Parametrization.EB)
    key_cmb_e, key_cmb_b, key_dust_e, key_dust_b, key_noise = jax.random.split(jax.random.PRNGKey(master_seed), 5)
    cmb_field = draw_field(field_like, field_covariance, key_cmb_e, key_cmb_b)
    noise = draw_noise(field_like, noise_covariance, key_noise)

    #the dust at the reference frequency
    if dust_model == PYSM_DUST:
        dust_covariance = None
        dust_field = pysm_dust_field(field_like, pysm_preset)
    else:
        dust_covariance = dust_covariance_operator_like(dust_amplitude_ee, dust_knee_ee,
                                                        dust_amplitude_bb, dust_knee_bb,
                                                        scalar_covariance)
        dust_field = draw_field(field_like, dust_covariance, key_dust_e, key_dust_b)

    frequency_scaling = dust_frequency_scaling(frequencies, frequency_exponent, dust_temperature)
    data = data_model(cmb_field, dust_field, frequency_scaling, mask, beam) + noise

    return {"data": data, "cmb_field": cmb_field, "dust_field": dust_field,
            "noise_covariance": noise_covariance, "scalar_covariance": scalar_covariance,
            "tensor_covariance": tensor_covariance, "field_covariance": field_covariance,
            "dust_covariance": dust_covariance, "mask": mask, "beam": beam,
            "frequencies": frequencies, "frequency_scaling": frequency_scaling,
            "dust_model": dust_model}
