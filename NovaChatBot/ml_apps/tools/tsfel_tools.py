from .processing_tools_final import chunk_data, mean_norm, loadfeaturesbydomain_sub
from . import multiprocess_tools as mpt

import numpy as np


def tsfelMulti(or_signal, signal, fs, win_size, overlap_size, features):
    feat_results = {}
    feat_dict = dict(temp_centroid="mpt.calc_centroid(signal, fs)",
                     temp_centroid_cum="mpt.calc_centroid_cum(signal, fs)",
                     temp_maxpks="mpt.maxpeaks(signal)",
                     temp_minpks="mpt.minpeaks(signal)",
                     temp_meandiff="mpt.mean_diff(signal)",
                     temp_mean_absdiff="mpt.mean_abs_diff(signal)",
                     temp_median_absdiff="mpt.median_abs_diff(signal)",
                     temp_dist="mpt.distance(signal)",
                     temp_sumabsdiff="mpt.sum_abs_diff(signal)",
                     temp_zcr="mpt.zero_cross(signal)",
                     temp_zcr_m="mpt.zero_cross_mean_rem(signal)",
                     temp_diff_zcr_m="mpt.diff_zero_cross_mean_rem(signal)",
                     temp_tenergy="mpt.total_energy(signal, fs)",
                     temp_slope="mpt.slope(signal)",
                     temp_auc="mpt.corrected_auc(signal, fs)",
                     temp_abs_energy="mpt.abs_energy(signal)",
                     temp_pk2pk="mpt.pk_pk_distance(signal)",
                     temp_entropy="mpt.entropy(signal)",
                     stat_interq="mpt.interq_range(signal)",
                     stat_kurt="mpt.kurtosis(signal)",
                     stat_ske="mpt.skewness(signal)",
                     stat_c_max="mpt.calc_max(signal)",
                     stat_c_min="mpt.calc_min(signal)",
                     stat_c_maxmin_diff="mpt.calc_maxmin_diff(signal)",
                     stat_c_min_m_rem="mpt.calc_min_mean_rem(signal)",
                     stat_c_max_m_rem="mpt.calc_max_mean_rem(signal)",
                     stat_c_mean="mpt.calc_mean(signal)",
                     stat_c_median="mpt.calc_median(signal)",
                     stat_c_std="mpt.calc_std(signal)",
                     stat_c_var="mpt.calc_var(signal)",
                     stat_m_abs_dev="mpt.mean_abs_deviation(signal)",
                     stat_med_abs_dev="mpt.median_abs_deviation(signal)",
                     stat_rms_s="mpt.rms(signal)",
                     stat_rms_s_mrem="mpt.rms_mean_rem(signal)",
                     spec_s_dist="mpt.spectral_distance(signal, fs=fs)",
                     spec_f_f="mpt.fundamental_frequency(signal, fs=fs)",
                     spec_m_freq="mpt.max_frequency(signal, fs)",
                     spec_kurt="mpt.spectral_kurtosis(signal, fs)",
                     spec_skew="mpt.spectral_skewness(signal, fs)",
                     spec_spread="mpt.spectral_spread(signal, fs)",
                     spec_roff="mpt.spectral_roll_off(signal, fs)",
                     spec_ron="mpt.spectral_roll_on(signal, fs)",
                     spec_entropy="mpt.spectral_entropy(signal, fs)",
                     spec_m_coeff="mpt.fft_mean_coeff(signal, fs)",
                     spec_m_flux="mpt.MeanSpectralFlux(or_signal, fs, win_size)",
                     spec_cumsum_spectogram="mpt.CumSumSpectogram(ftSxx)",
                     spec_max_spectogram="mpt.MaxSpectogram(ftSxx)",
                     spec_mean_spectogram="mpt.MeanSpectogram(ftSxx)",
                     spec_spectral_flux="mpt.SpecSpectralFlux(ftSxx)",
                     spec_wav_entropy="mpt.wavelet_entropy(signal)",
                     spec_power_band="mpt.power_bandwidth(signal, fs)",
                     spec_human_r_energy="mpt.human_range_energy(signal, fs)",
                     spec_max_pks="mpt.spectral_maxpeaks(signal, fs)",
                     spec_var="mpt.spectral_variation(signal, fs)",
                     spec_slope="mpt.spectral_slope(signal, fs)",
                     spec_decrease="mpt.spectral_decrease(signal, fs)",
                     spec_centroid="mpt.spectral_centroid(signal, fs)",
                     spec_median_f="mpt.median_frequency(signal, fs)",
                     spec_max_p_spec="mpt.max_power_spectrum(signal, fs)")

    # get Spectogram
    spec = [1 if "spectogram" in feat else 0 for feat in features]

    if (np.sum(spec) > 0):
        ftSxx = mpt.Spectrogram(signal, f_s=fs, win_size=win_size, overlap_size=overlap_size)
    # ftSxx = Spectrogram(signal, f_s=fs, win_size=win_size, overlap_size=overlap_size)
    for feature in features:
        feat_results[feature] = eval(feat_dict[feature])

    return feat_results


def featuresTsfelMat(inputSignal, fs, window_len, overlap_size, features):
    """
    :param inputSignal:1D signal or ND signal matrix
    :param fs:
    :param window_len:
    :param overlap_size:
    :return: matrix of features
    """

    # inputSignal = inputSignal - np.mean(inputSignal)
    WinRange = int(window_len / 2)

    if (np.ndim(inputSignal) > 1):

        # additionally added  iloc while getting error with my dataset
        t = np.linspace(0, len(inputSignal.iloc[:, 0]) / fs, len(inputSignal.iloc[:, 0]))
        t = chunk_data(np.r_[t[WinRange:0:-1], t, t[-1:len(t) - WinRange:-1]], window_size=window_len,
                       overlap_size=overlap_size)
        s_matrix = []
        # Corrected slicing using .iloc to reverse the DataFrame rows
        # and np.r_ to concatenate them.
        sig = np.r_[
            inputSignal.iloc[WinRange:0:-1].values,  # reversed initial window
            inputSignal.values,  # all rows
            inputSignal.iloc[-1:len(inputSignal) - WinRange:-1].values  # reversed final window
        ].transpose()

        for s_i in range(np.shape(sig)[0]):
            s_t = sig[s_i]
            s_temp = np.copy(s_t)
            # sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=overlap_size)*(win/win.sum())
            sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=overlap_size)
            s_matrix.append(sig_a)

        output = np.array(
            [{"signal": in_sig_i, "features": tsfelMulti(s_t, sig_i, fs, window_len, overlap_size, features)} for
             sig_i, in_sig_i in zip(s_matrix, inputSignal)])
    else:

        t = np.linspace(0, len(inputSignal) / fs, len(inputSignal))
        t = chunk_data(np.r_[t[WinRange:0:-1], t, t[-1:len(t) - WinRange:-1]], window_size=window_len,
                       overlap_size=overlap_size)
        sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]
        s_temp = np.copy(sig)
        # sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=overlap_size)*(win/win.sum())
        sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=overlap_size)

        output = np.array(
            [{"signal": inputSignal, "features": tsfelMulti(sig, sig_a, fs, window_len, overlap_size, features)}])

    return output


def load_featuresbydomain(file, features_tag="all"):
    """

    :param file: dictionnary of features
    :param features_tag: can be:
        "all"
        "tag: "temp, stat or spec"
        "specific feature name"
    :return: In the first 2 cases, extracts the feature matrix and the feature names array
    """
    if (features_tag == "all"):
        featureSet = dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])
        featureSet_names = dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])
        featureSet, feature_names = loadfeaturesbydomain_sub(file, featureSet, featureSet_names)

        return featureSet, feature_names
    elif (features_tag in ["temp", "stat", "spec"]):
        feature_array = []
        feature_names = []
        for feature in file.keys():
            if (feature in ["spec_m_coeff", "temp_mslope"]):
                continue
            elif (len(np.where(np.isnan(file[feature]))[0]) > 0):
                continue
            elif (features_tag in feature):
                signal_i = file[feature]

                # signal_i = mean_norm(signal_i)

                feature_array.append(signal_i)
                feature_names.append(features_tag)

        return feature_array, feature_names
    else:
        return file[features_tag]


def loadfeaturesbydomain_sub(features, featureSet, featureSet_names):
    for feature in features.keys():

        # print(np.where(np.isnan(features["features"][feature])))
        if (feature in ["spec_m_coeff", "temp_mslope"]):
            continue
        elif (len(np.where(np.isnan(np.array(features[feature])))[0]) > 0):
            # print(feature)
            continue
        elif (np.sum(abs(features[feature])) == 0):
            # print(feature)
            continue
        else:
            # print(feature)
            # print(len(features["features"][feature]))
            signal_i = features[feature]
            signal_i = mean_norm(signal_i)

            featureSet['allfeatures'].append(signal_i)
            featureSet_names["allfeatures"].append(feature)
            if ("temp" in feature):
                featureSet['featurebydomain']["temp"].append(signal_i)
                featureSet_names['featurebydomain']["temp"].append(feature)
            elif ("spec" in feature):
                featureSet['featurebydomain']["spec"].append(signal_i)
                featureSet_names['featurebydomain']["spec"].append(feature)
            else:
                featureSet['featurebydomain']["stat"].append(signal_i)
                featureSet_names['featurebydomain']["stat"].append(feature)
    return featureSet, featureSet_names
