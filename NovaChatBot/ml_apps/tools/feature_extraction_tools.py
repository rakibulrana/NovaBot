from .tsfel_tools import load_featuresbydomain, featuresTsfelMat
from .load_tools import load_Json
import numpy as np

def ExtractFeatureMatrix(or_signal, win_size, perc_overlap=0.9): # Why overlap value is fixed here
    """
    Computes the extraction of featues of a signal or a group of signals
    :param or_signal: signal or signals from which features will be extracted
    :param fs: sampling frequency
    :param time_scale: time scale at which features will be extracted. It defines the sliding window size, which is half of the time scale times the sampling frequency
    :param perc_overlap: overlap percentage of the sliding window
    :return: Feature matrix, feature dataframe with features by name and group, and sampling frequency of the extracted features
    """

    # temporal adjustments
    win_len = win_size // 2    # **** Why window size divided by 2??? ****

    # set features to extract
    def extract_selected_features(json_data, selected_features):
        """
        We add this method  since the original method doesn't have multiple selection of features options.
        Just to give the user to select features by their own choices
        :param json_data: mean, median, std etc. names
        :param selected_features: Selected features by users
        :return:
        """
        # Since json_data is a list of strings (feature names), filter this list
        extracted_features = [feature for feature in json_data if feature in selected_features]
        return extracted_features

    # Load features from JSON
    features = load_Json(r"ml_apps/tools/config1.json")["features"]
    selected_features = ["temp_sumabsdif", "temp_centroid_cum","temp_abs_energy", "temp_centroid", "temp_centroid_cum", "temp_maxpks", "temp_minpks"]

    # Extract the specific features
    extracted_features = extract_selected_features(features, selected_features)

    # extract features
    feat_file, featMat, feature_names = featuresExtraction(or_signal, 1, int(win_len), int(perc_overlap * win_len),
                                                           extracted_features)
    return featMat


def featuresExtraction(signal, fs, win_size, overlap_size, features): # why fs = 1?
    """
    Process of extracting features with methods from tsfel. It returns two dictionnarues with
    1) the feature file where the original signal and all the feature components are stored
    with the name of the features
    2) The feature name dictionnary where the tag of each feature is stored

    :param signal:  Original signal(s) from which the features will be extracted
    :param fs: sampling frequency (int)
    :param win_size: size of the sliding window
    :param overlap_size: overlaping size, if int, the value, if between 0-1, the percentage
    :return: 2 dictionnaries:
    1 - feature_file: Array with dicts for each signal from which features are extracted
    np.array(
			[{"signal": original signal, "features": matrix with features}])
	2 - feature_dict:
	dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])
	3 - feature_names:
	dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])

	TODO: Not yet consolidated the multisignal purposes
    """
    feature_file = featuresTsfelMat(signal, fs, win_size, overlap_size, features)

    if (np.ndim(signal) > 1):                       # How these works- work flow!! , features, all, allfeatures, etc.
        #first case
        feature_dict, featuredict_names = load_featuresbydomain(feature_file[0]["features"], "all")
        feature_dict={"allfeatures":feature_dict["allfeatures"]}
        featuredict_names={"allfeatures":featuredict_names["allfeatures"]}
        for i in range(0, len(feature_file)):
            #print("here is feature fliles: ",feature_file[i])
            feature_dict_, featuredict_names_ = load_featuresbydomain(feature_file[i]["features"], "all")
            # print(np.shape(feature_dict["allfeatures"]))
            feature_dict["allfeatures"] = np.vstack([feature_dict["allfeatures"], feature_dict_["allfeatures"]])
            featuredict_names["allfeatures"] = np.hstack([featuredict_names["allfeatures"], featuredict_names_["allfeatures"]])
    else:
        feature_dict, featuredict_names = load_featuresbydomain(feature_file[0]["features"], "all")
    return feature_file, feature_dict, featuredict_names