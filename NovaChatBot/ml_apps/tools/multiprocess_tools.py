import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import spectrogram
from tsfel.feature_extraction.features_utils import *
import sys
import numpy as np


# ############################################# TEMPORAL DOMAIN ##################################################### #





@set_domain("domain", "temporal")
def autocorr(signal):
    """Computes autocorrelation of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    float
        Cross correlation of 1-dimensional sequence

    """

    return np.correlate(signal, signal).astype(float)


@set_domain("domain", "temporal")
def calc_centroid(signal, fs):
    """Computes the centroid along the time axis.

    Parameters
    ----------
    signal : nd-array
        Input from which centroid is computed
    fs: int
        Signal sampling frequency

    Returns
    -------
    float
        Temporal centroid

    """

    # time = compute_time(signal[0], fs)
    time = np.repeat([np.linspace(0, len(signal[0]) / fs, len(signal[0]))], len(signal), axis=0)

    energy = signal ** 2

    t_energy = np.sum(np.multiply(time, energy), axis=1)
    energy_sum = np.sum(energy, axis=1)

    centroid = np.where(np.multiply(t_energy, energy_sum) == 0, 0, t_energy / energy_sum)

    return centroid


@set_domain("domain", "temporal")
def calc_centroid_cum(signal, fs):
    """Computes the centroid along the time axis.

    Parameters
    ----------
    signal : nd-array
        Input from which centroid is computed
    fs: int
        Signal sampling frequency

    Returns
    -------
    float
        Temporal centroid

    """

    # time = compute_time(signal[0], fs)
    centroid = calc_centroid(signal, fs)

    centroid = centroid - np.mean(centroid)

    return np.cumsum(centroid)


@set_domain("domain", "temporal")
def minpeaks(signal):
    """Computes number of minimum peaks of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which minimum number of peaks is counted
    Returns
    -------
    float
        Minimum number of peaks

    """
    thr = np.std(signal)
    meta_signal = np.where(np.abs(signal) > 0.5 * thr, signal, 0)

    diff_sig = np.diff(meta_signal)

    min_pks = np.where(diff_sig < 0, 3, 0)

    return np.sum(np.where(np.diff(min_pks) == -3, 1, 0), axis=1)


@set_domain("domain", "temporal")
def maxpeaks(signal):
    """Computes number of maximum peaks of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which maximum number of peaks is counted

    Returns
    -------
    float
        Maximum number of peaks

    """
    thr = np.std(signal)

    meta_signal = np.where(np.abs(signal) > 0.5 * thr, signal, 0)

    diff_sig = np.diff(meta_signal)

    max_pks = np.where(diff_sig > 0, 3, 0)

    return np.sum(np.where(np.diff(max_pks) == 3, 1, 0), axis=1)


@set_domain("domain", "temporal")
def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which mean absolute deviation is computed

   Returns
   -------
   float
       Mean absolute difference result

   """

    return np.mean(np.abs(np.diff(signal)), axis=1)


@set_domain("domain", "temporal")
def mean_diff(signal):
    """Computes mean of differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which mean of differences is computed

   Returns
   -------
   float
       Mean difference result

   """

    return np.mean(np.diff(signal), axis=1)


@set_domain("domain", "temporal")
def median_abs_diff(signal):
    """Computes median absolute differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which median absolute difference is computed

   Returns
   -------
   float
       Median absolute difference result

   """

    return np.median(np.abs(np.diff(signal)), axis=1)


@set_domain("domain", "temporal")
def median_diff(signal):
    """Computes median of differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which median of differences is computed

   Returns
   -------
   float
       Median difference result

   """

    return np.median(np.diff(signal), axis=1)


@set_domain("domain", "temporal")
def distance(signal):
    """Computes signal traveled distance.

    Calculates the total distance traveled by the signal
    using the hipotenusa between 2 datapoints.

    Parameters
    ----------
    signal : nd-array
        Input from which distance is computed

    Returns
    -------
    float
        Signal distance

    """
    diff_sig = np.diff(signal)
    return np.sum(np.sqrt(1 + diff_sig ** 2), axis=1)


@set_domain("domain", "temporal")
def sum_abs_diff(signal):
    """Computes sum of absolute differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which sum absolute difference is computed

   Returns
   -------
   float
       Sum absolute difference result

   """

    return np.sum(np.abs(np.diff(signal)), axis=1)


@set_domain("domain", "temporal")
def zero_cross(signal):
    """Computes Zero-crossing rate of the signal.

    Corresponds to the total number of times that the signal changes from
    positive to negative or vice versa.

    Parameters
    ----------
    signal : nd-array
        Input from which the zero-crossing rate are computed

    Returns
    -------
    int
        Number of times that signal value cross the zero axis

    """

    return np.sum(np.where(np.diff(np.sign(signal)) != 0, 1, 0), axis=1)


@set_domain("domain", "temporal")
def zero_cross_mean_rem(signal):
    """Computes Zero-crossing rate of the signal.

    Corresponds to the total number of times that the signal changes from
    positive to negative or vice versa.

    Parameters
    ----------
    signal : nd-array
        Input from which the zero-crossing rate are computed

    Returns
    -------
    int
        Number of times that signal value cross the zero axis after removing the mean

    """

    sig = np.array((signal.T - np.mean(signal, axis=1))).T

    return np.sum(np.where(np.diff(np.sign(sig)) != 0, 1, 0), axis=1)


@set_domain("domain", "temporal")
def diff_zero_cross_mean_rem(signal):
    """Computes Zero-crossing rate of the signal.

    Corresponds to the total number of times that the signal changes from
    positive to negative or vice versa.

    Parameters
    ----------
    signal : nd-array
        Input from which the zero-crossing rate are computed

    Returns
    -------
    int
        Number of times that signal value cross the zero axis after removing the mean

    """

    sig = np.array((signal.T - np.mean(signal, axis=1))).T

    return np.diff(np.sum(np.where(np.diff(np.sign(sig)) != 0, 1, 0), axis=1))


@set_domain("domain", "temporal")
def total_energy(signal, fs):
    """Computes the total energy of the signal.

    Parameters
    ----------
    signal : nd-array
        Signal from which total energy is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Total energy

    """

    return np.sum(np.array(signal) ** 2, axis=1) / (len(signal[0]) / fs)


@set_domain("domain", "temporal")
def slope(signal):
    """Computes the slope of the signal.

    Slope is computed by fitting a linear equation to the observed data.

    Parameters
    ----------
    signal : nd-array
        Input from which linear equation is computed

    Returns
    -------
    float
        Slope

    NOTA: it is still slow because of the for loop...
    """
    t = np.linspace(0, len(signal[0]) - 1, len(signal[0]))

    return np.array([np.polyfit(t, signal_i, 1)[0] for signal_i in signal])


@set_domain("domain", "temporal")
def auc(signal, fs):
    """Computes the area under the curve of the signal computed with trapezoid rule.

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed
    fs : int
        Sampling Frequency
    Returns
    -------
    float
        The area under the curve value

    """
    t = [1 / fs] * len(signal[0])
    s1 = signal[:, :-1]
    s2 = signal[:, 1:]
    trap_area = t[1:] * (s2 + s1) / 2
    return np.sum(trap_area, axis=1)


@set_domain("domain", "temporal")
def corrected_auc(signal, fs):
    """Computes the area under the curve of the signal computed with trapezoid rule.

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed
    fs : int
        Sampling Frequency
    Returns
    -------
    float
        The area under the curve value

    """
    auc_ = auc(signal, fs)

    return -auc_


@set_domain("domain", "temporal")
def abs_energy(signal):
    """Computes the absolute energy of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        Absolute energy

    """

    return np.sum(signal ** 2, axis=1)


@set_domain("domain", "temporal")
def pk_pk_distance(signal):
    """Computes the peak to peak distance.

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        peak to peak distance

    """

    return np.abs(np.max(signal, axis=1) - np.min(signal, axis=1))


@set_domain("domain", "temporal")
def entropy(signal, prob='kde'):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    float
        The normalized entropy value

    """
    output = np.zeros(len(signal))
    for j, signal_i in enumerate(signal):

        if prob == 'kde':
            p = kde(signal_i)
        elif prob == 'gauss':
            p = gaussian(signal)

        if np.sum(p) == 0:
            return 0.0

        # Handling zero probability values
        p = p[np.where(p != 0)]

        if np.sum(p * np.log2(p)) / np.log2(len(signal_i)) == 0:
            continue
        else:
            output[j] = - np.sum(p * np.log2(p)) / np.log2(len(signal_i))

    return output


# ############################################ STATISTICAL DOMAIN #################################################### #

# NOTA: Relativamente ao histograma seria possivel olhar para ele como uma outra dimensao do sinal e perceber que features
# extrair a partir dai...nomeadamente, o numero de picos no histograma, que zonas de densidade, quantidade de bins em modo "auto"

@set_domain("domain", "statistical")
def hist(signal, nbins=10, r=1):
    """Computes histogram of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from histogram is computed
    nbins : int
        The number of equal-width bins in the given range
    r : float
        The lower(-r) and upper(r) range of the bins

    Returns
    -------
    nd-array
        The values of the histogram

    """

    histsig, bin_edges = np.histogram(signal, bins=nbins, range=[-r, r])  # TODO:subsampling parameter

    return tuple(histsig)


@set_domain("domain", "statistical")
def interq_range(signal):
    """Computes interquartile range of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which interquartile range is computed

    Returns
    -------
    float
        Interquartile range result

    """

    return np.percentile(signal, 75, axis=1) - np.percentile(signal, 25, axis=1)


@set_domain("domain", "statistical")
def kurtosis(signal):
    """Computes kurtosis of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which kurtosis is computed

    Returns
    -------
    float
        Kurtosis result

    """

    return scipy.stats.kurtosis(signal, axis=1)


@set_domain("domain", "statistical")
def skewness(signal):
    """Computes skewness of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which skewness is computed

    Returns
    -------
    int
        Skewness result

    """

    return scipy.stats.skew(signal, axis=1)


@set_domain("domain", "statistical")
def calc_maxmin_diff(signal):
    """Computes the maximum value of the signal.

    Parameters
    ----------
    signal : nd-array
       Input from which max is computed

    Returns
    -------
    float
        Maximum result

    """

    sig_diff = np.max(signal, axis=1) - np.min(signal, axis=1)

    return sig_diff


@set_domain("domain", "statistical")
def calc_max(signal):
    """Computes the maximum value of the signal.

    Parameters
    ----------
    signal : nd-array
       Input from which max is computed

    Returns
    -------
    float
        Maximum result

    """

    return np.max(signal, axis=1)


@set_domain("domain", "statistical")
def calc_max_mean_rem(signal):
    """Computes the minimum value of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which min is computed

    Returns
    -------
    float
        Minimum result

    """
    sig = np.array((signal.T - np.mean(signal, axis=1))).T
    return np.abs(np.max(sig, axis=1))


@set_domain("domain", "statistical")
def calc_min(signal):
    """Computes the minimum value of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which min is computed

    Returns
    -------
    float
        Minimum result

    """

    return np.min(signal, axis=1)


@set_domain("domain", "statistical")
def calc_min_mean_rem(signal):
    """Computes the minimum value of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which min is computed

    Returns
    -------
    float
        Minimum result

    """
    sig = np.array((signal.T - np.mean(signal, axis=1))).T
    return np.abs(np.min(sig, axis=1))


@set_domain("domain", "statistical")
def calc_mean(signal):
    """Computes mean value of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which mean is computed.

    Returns
    -------
    float
        Mean result

    """

    return np.mean(signal, axis=1)


@set_domain("domain", "statistical")
def calc_median(signal):
    """Computes median of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which median is computed

    Returns
    -------
    float
        Median result

    """
    return np.median(signal, axis=1)


@set_domain("domain", "statistical")
def mean_abs_deviation(signal):
    """Computes mean absolute deviation of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result

    """
    sig = np.abs(np.array((signal.T - np.mean(signal, axis=1))).T)

    return np.mean(sig, axis=1)


@set_domain("domain", "statistical")
def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result

    """

    return scipy.stats.median_abs_deviation(signal, scale=1, axis=1)


@set_domain("domain", "statistical")
def rms(signal):
    """Computes root mean square of the signal.

    Square root of the arithmetic mean (average) of the squares of the original values.

    Parameters
    ----------
    signal : nd-array
        Input from which root mean square is computed

    Returns
    -------
    float
        Root mean square

    """

    return np.sqrt(np.sum(np.array(signal) ** 2, axis=1) / len(signal[0]))


@set_domain("domain", "statistical")
def rms_mean_rem(signal):
    """Computes root mean square of the signal.

    Square root of the arithmetic mean (average) of the squares of the original values.

    Parameters
    ----------
    signal : nd-array
        Input from which root mean square is computed

    Returns
    -------
    float
        Root mean square

    """
    sig = np.array((signal.T - np.mean(signal, axis=1))).T

    return np.sqrt(np.sum(np.array(sig) ** 2, axis=1) / len(signal[0]))


@set_domain("domain", "statistical")
def calc_std(signal):
    """Computes standard deviation (std) of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which std is computed

    Returns
    -------
    float
        Standard deviation result

    """

    return np.std(signal, axis=1)


@set_domain("domain", "statistical")
def calc_var(signal):
    """Computes variance of the signal.

    Parameters
    ----------
    signal : nd-array
       Input from which var is computed

    Returns
    -------
    float
        Variance result

    """

    return np.var(signal, axis=1)


# @set_domain("domain", "statistical")
# def ecdf(signal, d=10):
#     """Computes the values of ECDF (empirical cumulative distribution function) along the time axis.
#
#     Parameters
#     ----------
#     signal : nd-array
#         Input from which ECDF is computed
#     d: integer
#         Number of ECDF values to return
#
#     Returns
#     -------
#     float
#         The values of the ECDF along the time axis
#     """
#     sorted_sig, y = np.sort(signal), np.arange(1, len(signal[0]) + 1) / len(signal[0])
#     # _, y = calc_ecdf(signal)
#     if len(signal[0]) <= d:
#         return tuple(y)
#     else:
#         return tuple(y[:d])
#
#
# @set_domain("domain", "statistical")
# def ecdf_slope(signal, p_init=0.5, p_end=0.75):
#     """Computes the slope of the ECDF between two percentiles.
#
#     Parameters
#     ----------
#     signal : nd-array
#         Input from which ECDF is computed
#     p_init : float
#         Initial percentile
#     p_end : float
#         End percentile
#
#     Returns
#     -------
#     float
#         The slope of the ECDF between two percentiles
#     """
#
#     sumdiff = np.sum(np.diff(signal), axis=1)
#
#     x_init, x_end = ecdf_percentile(signal, percentile=[p_init, p_end])
#
#     sumdiff = np.where(sumdiff == 0, np.inf, sumdiff)
#
#     else:
#
#         return (p_end - p_init) / (x_end - x_init)
#
#
# @set_domain("domain", "statistical")
# def ecdf_percentile(signal, percentile=None):
#     """Determines the percentile value of the ECDF.
#
#     Parameters
#     ----------
#     signal : nd-array
#         Input from which ECDF is computed
#     percentile: list
#         Percentile value to be computed
#
#     Returns
#     -------
#     float
#         The input value(s) of the ECDF
#     """
#     if percentile is None:
#         percentile = [0.2, 0.8]
#     if type(percentile) in [float, int]:
#         percentile = [percentile]
#
#     # calculate ecdf
#     x, y = np.sort(signal), np.arange(1, len(signal[0]) + 1) / len(signal[0])
#
#     if len(percentile) > 1:
#         sumdiff = np.sum(np.diff(signal), axis=1)
#
#         # check if signal is constant
#         if np.sum(np.diff(signal),axis=1) == 0:
#             return tuple(np.repeat(signal[0], len(percentile)))
#         else:
#             return tuple([x[y <= p].max() for p in percentile])
#     else:
#         # check if signal is constant
#         if np.sum(np.diff(signal)) == 0:
#             return signal[0]
#         else:
#             return x[y <= percentile].max()
#
#
# @set_domain("domain", "statistical")
# def ecdf_percentile_count(signal, percentile=None):
#     """Determines the cumulative sum of samples that are less than the percentile.
#
#     Parameters
#     ----------
#     signal : nd-array
#         Input from which ECDF is computed
#     percentile: list
#         Percentile threshold
#
#     Returns
#     -------
#     float
#         The cumulative sum of samples
#     """
#     if percentile is None:
#         percentile = [0.2, 0.8]
#     if type(percentile) in [float, int]:
#         percentile = [percentile]
#
#     # calculate ecdf
#     x, y = calc_ecdf(signal)
#
#     if len(percentile) > 1:
#         # check if signal is constant
#         if np.sum(np.diff(signal)) == 0:
#             return tuple(np.repeat(signal[0], len(percentile)))
#         else:
#             return tuple([x[y <= p].shape[0] for p in percentile])
#     else:
#         # check if signal is constant
#         if np.sum(np.diff(signal)) == 0:
#             return signal[0]
#         else:
#             return x[y <= percentile].shape[0]
#

# ############################################## SPECTRAL DOMAIN ##################################################### #

@set_domain("domain", "spectral")
def spectral_distance(signal, fs):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        spectral distance

    """

    f, fmag = calc_fftmulti(signal, fs)

    cum_fmag = np.cumsum(fmag, axis=1)

    # Computing the linear regression
    points_y = np.array([np.linspace(0, cum_fmag_i[-1], np.shape(cum_fmag)[1]) for cum_fmag_i in cum_fmag])

    return np.sum(points_y - cum_fmag, axis=1)


@set_domain("domain", "spectral")
def spectral_distance_corrected(signal, fs):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        spectral distance

    """

    f, fmag = calc_fftmulti(signal, fs)

    cum_fmag = np.cumsum(fmag, axis=1)

    # Computing the linear regression
    points_y = np.array([np.linspace(0, cum_fmag_i[-1], np.shape(cum_fmag)[1]) for cum_fmag_i in cum_fmag])

    return smooth(np.sum(points_y - cum_fmag, axis=1), window_len=len(signal[0] / 10))


@set_domain("domain", "spectral")
def corrected_spectral_distance(signal, fs):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        spectral distance

    """

    sp_dist = spectral_distance(signal, fs)

    return -sp_dist


def calc_fftmulti(signal, fs):
    """ This functions computes the fft of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    fs : int
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)

    """

    fmag = np.abs(np.fft.fft(signal, axis=1))
    f = np.linspace(0, fs // 2, len(signal[0]) // 2)

    return f[:len(signal[0]) // 2].copy(), fmag[:, :len(signal[0]) // 2].copy()


@set_domain("domain", "spectral")
def fundamental_frequency(signal_windows, fs):
    """Computes fundamental frequency of the signal.

    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.

    Parameters
    ----------
    signal : nd-array
        Input from which fundamental frequency is computed
    fs : int
        Sampling frequency

    Returns
    -------
    f0: float
       Predominant frequency of the signal

    """

    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        signal = signal - np.mean(signal)
        f, fmag = calc_fft(signal, fs)

        # Condition for offset removal, since the offset generates a peak at frequency zero
        try:
            # With 0.1 the offset frequency is discarded
            cond = np.where(f > 0.1)[0][0]
        except IndexError:
            cond = 0

        # Finding big peaks, not considering noise peaks with low amplitude
        bp = scipy.signal.find_peaks(fmag[cond:], threshold=10)[0]
        if not list(bp):
            f0 = 0
        else:
            # add cond for correcting the indices position
            bp = bp + cond
            # f0 is the minimum big peak frequency
            f0 = f[min(bp)]

        output[ind] = f0
    return output


@set_domain("domain", "spectral")
def max_power_spectrum(signal_windows, fs):
    """Computes maximum power spectrum density of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which maximum power spectrum is computed
    fs : scalar
        Sampling frequency

    Returns
    -------
    nd-array
        Max value of the power spectrum density

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        if np.std(signal) == 0:
            output[ind] = float(max(scipy.signal.welch(signal, int(fs), nperseg=len(signal))[1]))
        else:
            output[ind] = float(max(scipy.signal.welch(signal / np.std(signal), int(fs), nperseg=len(signal))[1]))

    return output


@set_domain("domain", "spectral")
def max_frequency(signal, fs):
    """Computes maximum frequency of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which maximum frequency is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
    signal = signal - np.repeat([np.mean(signal, axis=1)], len(signal[0]), axis=0).T
    f, fmag = calc_fftmulti(signal, fs)
    cum_fmag = np.cumsum(fmag, axis=1)
    np.shape(cum_fmag)

    c_sub = cum_fmag[:, -1] * 0.95
    c_sub = np.reshape(c_sub, (len(c_sub), 1))
    c_ss = np.diff(np.sign(cum_fmag - c_sub), axis=1)

    ind_mag = np.argmax(cum_fmag, axis=1) * np.ones(np.shape(signal)[0]).astype(int)

    t_ind = np.where(c_ss == 2)
    ind_mag[t_ind[0]] = t_ind[1]

    #
    # try:
    #     ind_mag = np.where(c_ss==2, )
    #     print("hello")
    # except IndexError:
    #     ind_mag = np.argmax(cum_fmag, axis=0)
    #     print("astaa")

    return f[ind_mag]


@set_domain("domain", "spectral")
def median_frequency(signal_windows, fs):
    """Computes median frequency of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which median frequency is computed
    fs: int
        Sampling frequency

    Returns
    -------
    f_median : int
       0.50 of maximum frequency using cumsum.
    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        signal = signal - np.mean(signal)

        f, fmag = calc_fft(signal, fs)
        cum_fmag = np.cumsum(fmag)
        try:
            ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
        except IndexError:
            ind_mag = np.argmax(cum_fmag)
        # np.stack(np.nonzero(a == np.percentile(a, 50, interpolation='nearest')), axis=1)
        # ind_mag2 = np.nonzero(cum_fmag == 0.5*cum_fmag)
        f_median = f[ind_mag]
        output[ind] = f_median

    return output


@set_domain("domain", "spectral")
def spectral_centroid(signal_windows, fs):
    """Barycenter of the spectrum.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral centroid is computed
    fs: int
        Sampling frequency

    Returns
    -------
    float
        Centroid

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        f, fmag = calc_fft(signal, fs)
        if not np.sum(fmag):
            output[ind] = 0
        else:
            output[ind] = np.dot(f, fmag / np.sum(fmag))

    return output


@set_domain("domain", "spectral")
def spectral_decrease(signal_windows, fs):
    """Represents the amount of decreasing of the spectra amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral decrease is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral decrease

    """

    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        f, fmag = calc_fft(signal, fs)

        fmag_band = fmag[1:]
        len_fmag_band = np.arange(2, len(fmag) + 1)

        # Sum of numerator
        soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

        if not np.sum(fmag_band):
            output[ind] = 0
        else:
            # Sum of denominator
            soma_den = 1 / np.sum(fmag_band)
            output[ind] = soma_den * soma_num
    # Spectral decrease computing
    return output


@set_domain("domain", "spectral")
def spectral_kurtosis(signal, fs):
    """Measures the flatness of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral kurtosis is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Kurtosis

    """

    f, fmag = calc_fftmulti(signal, fs)
    s_c = spectral_centroid(signal, fs)
    num_sc = ((f - np.reshape(s_c, (len(s_c), 1))) ** 4)
    den_sc = fmag / np.reshape(np.sum(fmag, axis=1), (np.shape(fmag)[0], 1))
    spect_kurt = num_sc * (den_sc)
    return np.sum(spect_kurt, axis=1) / (spectral_spread(signal, fs) ** 4)


@set_domain("domain", "spectral")
def spectral_skewness(signal, fs):
    """Measures the asymmetry of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral skewness is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Skewness

    """

    f, fmag = calc_fftmulti(signal, fs)
    spect_centr = spectral_centroid(signal, fs)
    den_sc = fmag / np.reshape(np.sum(fmag, axis=1), (np.shape(fmag)[0], 1))

    skew = ((f - np.reshape(spect_centr, (len(spect_centr), 1))) ** 3) * (den_sc)
    return np.sum(skew, axis=1) / (spectral_spread(signal, fs) ** 3)


@set_domain("domain", "spectral")
def spectral_spread(signal, fs):
    """Measures the spread of the spectrum around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral spread is computed.
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Spread

    """

    f, fmag = calc_fftmulti(signal, fs)
    s_c = spectral_centroid(signal, fs)

    num_sc = (f - np.reshape(s_c, (len(s_c), 1))) ** 2
    den = fmag / np.reshape(np.sum(fmag, axis=1), (np.shape(fmag)[0], 1))
    s_s = np.sum(np.multiply(num_sc, den), axis=1) ** 0.5

    return s_s


@set_domain("domain", "spectral")
def spectral_slope(signal_windows, fs):
    """Computes the spectral slope.

    Spectral slope is computed by finding constants m and b of the function aFFT = mf + b, obtained by linear regression
    of the spectral amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral slope is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Slope

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        f, fmag = calc_fft(signal, fs)

        if not (list(f)) or (np.sum(fmag) == 0):
            output[ind] = 0
        else:
            if not (len(f) * np.dot(f, f) - np.sum(f) ** 2):
                output[ind] = 0
            else:
                num_ = (1 / np.sum(fmag)) * (len(f) * np.dot(f, fmag) - np.sum(f) * np.sum(fmag))
                denom_ = (len(f) * np.dot(f, f) - np.sum(f) ** 2)
                output[ind] = num_ / denom_
    return output


@set_domain("domain", "spectral")
def spectral_variation(signal_windows, fs):
    """Computes the amount of variation of the spectrum along time.

    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral variation is computed.
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Variation

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        f, fmag = calc_fft(signal, fs)

        sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
        sum2 = np.sum(np.array(fmag)[1:] ** 2)
        sum3 = np.sum(np.array(fmag)[:-1] ** 2)

        if not sum2 or not sum3:
            variation = 1
        else:
            variation = 1 - (sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5)))
        output[ind] = variation
    return output


@set_domain("domain", "spectral")
def spectral_maxpeaks(signal_windows, fs):
    """Computes number of maximum spectral peaks of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which the number of maximum spectral peaks is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Total number of maximum spectral peaks

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        f, fmag = calc_fft(signal, fs)
        diff_sig = np.diff(fmag)

        output[ind] = np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd + 1] < 0) and (diff_sig[nd] > 0))])
    return output


@set_domain("domain", "spectral")
def spectral_roll_off(signal, fs):
    """Computes the spectral roll-off of the signal.

    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained
    below of this value.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-off is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-off

    """

    f, fmag = calc_fftmulti(signal, fs)
    cum_ff = np.cumsum(fmag, axis=1)
    value = 0.95 * (np.sum(fmag, axis=1))
    a = np.where(cum_ff >= np.reshape(value, (len(value), 1)), 1, 0)
    s = np.sum(np.diff(a, axis=1), axis=1) - 1
    s[np.where(s == 0)[0]] = np.where(np.diff(a, axis=1) == 1)[1]

    return f[s + 1]


@set_domain("domain", "spectral")
def spectral_roll_on(signal, fs):
    """Computes the spectral roll-on of the signal.

    The spectral roll-on corresponds to the frequency where 5% of the signal magnitude is contained
    below of this value.

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-on is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-on

    """

    f, fmag = calc_fftmulti(signal, fs)
    cum_ff = np.cumsum(fmag, axis=1)
    value = 0.05 * (np.sum(fmag, axis=1))
    a = np.where(cum_ff >= np.reshape(value, (len(value), 1)), 1, 0)
    s = np.sum(np.diff(a, axis=1), axis=1) - 1
    s[np.where(s == 0)[0]] = np.where(np.diff(a, axis=1) == 1)[1]

    return f[s + 1]


@set_domain("domain", "spectral")
def human_range_energy(signal_windows, fs):
    """Computes the human range energy ratio.

    The human range energy ratio is given by the ratio between the energy
    in frequency 0.6-2.5Hz and the whole energy band.

    Parameters
    ----------
    signal : nd-array
        Signal from which human range energy ratio is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Human range energy ratio

    """

    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        f, fmag = calc_fft(signal, fs)

        allenergy = np.sum(fmag ** 2)

        if allenergy == 0:
            # For handling the occurrence of Nan values
            return 0.0

        hr_energy = np.sum(fmag[np.argmin(np.abs(0.6 - f)):np.argmin(np.abs(2.5 - f))] ** 2)

        ratio = hr_energy / allenergy
        output[ind] = ratio
    return output


#
# @set_domain("domain", "spectral")
# def mfcc(signal_windows, fs, pre_emphasis=0.97, nfft=512, nfilt=40, num_ceps=12, cep_lifter=22):
#     """Computes the MEL cepstral coefficients.
#
#     It provides the information about the power in each frequency band.
#
#     Implementation details and description on:
#     https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
#     https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1
#
#     Parameters
#     ----------
#     signal : nd-array
#         Input from which MEL coefficients is computed
#     fs : int
#         Sampling frequency
#     pre_emphasis : float
#         Pre-emphasis coefficient for pre-emphasis filter application
#     nfft : int
#         Number of points of fft
#     nfilt : int
#         Number of filters
#     num_ceps: int
#         Number of cepstral coefficients
#     cep_lifter: int
#         Filter length
#
#     Returns
#     -------
#     nd-array
#         MEL cepstral coefficients
#
#     """
#     output = np.empty(np.shape(signal_windows)[0], float)
#     for ind, signal in enumerate(signal_windows):
#         filter_banks = filterbank(signal, fs, pre_emphasis, nfft, nfilt)
#
#         mel_coeff = scipy.fft.dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_ceps + 1)]  # Keep 2-13
#
#         mel_coeff -= (np.mean(mel_coeff, axis=0) + 1e-8)
#
#         # liftering
#         ncoeff = len(mel_coeff)
#         n = np.arange(ncoeff)
#         lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)  # cep_lifter = 22 from python_speech_features library
#
#         mel_coeff *= lift
#
#
#     return tuple(mel_coeff)


@set_domain("domain", "spectral")
def power_bandwidth(signal_windows, fs):
    """Computes power spectrum density bandwidth of the signal.

    It corresponds to the width of the frequency band in which 95% of its power is located.

    Description in article:
    Power Spectrum and Bandwidth Ulf Henriksson, 2003 Translated by Mikael Olofsson, 2005

    Parameters
    ----------
    signal : nd-array
        Input from which the power bandwidth computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Occupied power in bandwidth

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        # Computing the power spectrum density
        if np.std(signal) == 0:
            freq, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
        else:
            freq, power = scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))

        if np.sum(power) == 0:
            return 0.0

        # Computing the lower and upper limits of power bandwidth
        cum_power = np.cumsum(power)
        f_lower = freq[np.where(cum_power >= cum_power[-1] * 0.95)[0][0]]

        cum_power_inv = np.cumsum(power[::-1])
        f_upper = freq[np.abs(np.where(cum_power_inv >= cum_power[-1] * 0.95)[0][0] - len(power) + 1)]

        # Returning the bandwidth in terms of frequency
        output[ind] = np.abs(f_upper - f_lower)
    return output


@set_domain("domain", "spectral")
def fft_mean_coeff(ftSxx):
    """Computes the mean value of each spectrogram frequency.

    nfreq can not be higher than half signal length plus one.
    When it does, it is automatically set to half signal length plus one.

    Parameters
    ----------
    signal : nd-array
        Input from which fft mean coefficients are computed
    fs : int
        Sampling frequency
    nfreq : int
        The number of frequencies

    Returns
    -------
    nd-array
        The mean value of each spectrogram frequency

    """
    f, t, X = ftSxx

    return np.mean(ftSxx, axis=1)


@set_domain("domain", "spectral")
def lpcc(signal, n_coeff=12):
    """Computes the linear prediction cepstral coefficients.

    Implementation details and description in:
    http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction cepstral coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction cepstral coefficients

    """

    # 12-20 cepstral coefficients are sufficient for speech recognition
    lpc_coeffs = lpc(signal, n_coeff)

    if np.sum(lpc_coeffs) == 0:
        return tuple(np.zeros(n_coeff))

    # Power spectrum
    powerspectrum = np.abs(np.fft.fft(lpc_coeffs)) ** 2
    lpcc_coeff = np.fft.ifft(np.log(powerspectrum))

    return tuple(np.abs(lpcc_coeff))


@set_domain("domain", "spectral")
def spectral_entropy(signal, fs):
    """Computes the spectral entropy of the signal based on Fourier transform.

    Parameters
    ----------
    signal : nd-array
        Input from which spectral entropy is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        The normalized spectral entropy value

    """

    # Removing DC component
    sig = signal - np.reshape(np.mean(signal, axis=1), (len(np.mean(signal, axis=1)), 1))

    f, fmag = calc_fftmulti(sig, fs)

    power = fmag ** 2
    power_s = np.sum(power, axis=1)

    prob = power / np.reshape(power_s, (len(power_s), 1))

    ind = np.where(prob != 0)
    prob = np.where(prob != 0, prob, sys.float_info.min)

    return -np.multiply(prob, np.log2(prob)).sum(axis=1) / np.log2(prob.shape[1])


@set_domain("domain", "spectral")
def wavelet_entropy(signal_windows, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT entropy of the signal.

    Implementation details in:
    https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy
    B.F. Yan, A. Miyamoto, E. Bruhwiler, Wavelet transform-based modal parameter identification considering uncertainty

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    float
        wavelet entropy

    """
    output = np.empty(np.shape(signal_windows)[0], float)
    for ind, signal in enumerate(signal_windows):
        if np.sum(signal) == 0:
            return 0.0

        cwt = wavelet(signal, function, widths)
        energy_scale = np.sum(np.abs(cwt), axis=1)
        t_energy = np.sum(energy_scale)
        prob = energy_scale / t_energy
        w_entropy = -np.sum(prob * np.log(prob))
        output[ind] = w_entropy
    return output


@set_domain("domain", "spectral")
def wavelet_abs_mean(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT absolute mean value of each wavelet scale.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT absolute mean value

    """

    return tuple(np.abs(np.mean(wavelet(signal, function, widths), axis=1)))


@set_domain("domain", "spectral")
def wavelet_std(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT std value of each wavelet scale.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT std

    """

    return tuple((np.std(wavelet(signal, function, widths), axis=1)))


@set_domain("domain", "spectral")
def wavelet_var(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT variance value of each wavelet scale.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT variance

    """

    return tuple((np.var(wavelet(signal, function, widths), axis=1)))


@set_domain("domain", "spectral")
def wavelet_energy(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT energy of each wavelet scale.

    Implementation details:
    https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT energy

    """

    cwt = wavelet(signal, function, widths)
    energy = np.sqrt(np.sum(cwt ** 2, axis=1) / np.shape(cwt)[1])

    return tuple(energy)


# @set_domain("domain", "spectral")
# def MeanSpectralFlux(signal_windows, fs, win_size):
#
#     output = np.empty(np.shape(signal_windows)[0], float)
#     for ind, signal in enumerate(signal_windows):
#
#         [f, t, X] = Spectrogram(signal, f_s=fs, win_size=win_size//2, overlap_size=win_size//4)
#
#         X = np.sqrt(X/2)
#
#         d_flux = SpectralFlux(X)
#
#         output[ind] = np.mean(d_flux)
#     return output

@set_domain("domain", "spectral")
def MeanSpectogram(ftSxx):
    f, t, X = ftSxx

    X = np.sqrt(X / 2)

    return np.mean(X, axis=0)


@set_domain("domain", "spectral")
def MaxSpectogram(ftSxx):
    f, t, X = ftSxx

    X = np.sqrt(X / 2)

    return np.max(X, axis=0)


def SpecSpectralFlux(ftSxx):
    f, t, X = ftSxx

    X = np.sqrt(X / 2)

    d_flux = SpectralFlux(X)

    return d_flux


def CumSumSpectogram(ftSxx):
    f, t, X = ftSxx

    X = np.sqrt(X / 2)

    return np.sum(X, axis=0)


def SpectralFlux(X):
    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    # X = np.concatenate(X[:,0],X, axis=1)
    afDeltaX = np.diff(X, 1, axis=1)

    # flux
    vsf = np.sqrt((afDeltaX ** 2).sum(axis=0)) / X.shape[0]

    return (np.r_[0, vsf])


def Spectrogram(signal, f_s, win_size, overlap_size):
    """
    Apply spetogram from scipy.
    :param signal: 1D signal
    :param f_s: sampling frequency
    :param win_size: For better results, choose win_size with power of 2. The longer the window, the better frequency resolution, but
    the worst time resolution.
    :param overlap_size: Overlap size between time segments to calculate the FFT
    :return:
    """
    f, t, Sxx = spectrogram(signal, f_s, nperseg=win_size, noverlap=overlap_size)
    # f, t, Sxx = spectrogram(signal, f_s)

    return f, t, Sxx

# def ImplementationofFourierTempogram():
#
