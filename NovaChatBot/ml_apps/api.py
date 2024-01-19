from rest_framework.decorators import api_view
from django.http import JsonResponse
import json
import pandas as pd
import numpy as np
from .tools.ssm_tools import compute_ssm, compute_novelty


@api_view(['GET', 'POST'])
def my_api_view(request):
    # Make sure you have the correct method
    if request.method == 'POST':
        # Parse the body of the request to JSON
        received_data = json.loads(request.body)

        # Access the data using the dictionary key
        zoomed_channel_data = received_data.get('zoomedChannelData')
        selected_features = received_data.get('selectedFeatures')
        window_length = int(received_data.get('windowLength', 0))
        window_overlap = int(received_data.get('windowOverlap', 0))
        features_list = [feature for category in selected_features.values() for feature in category]


        if zoomed_channel_data and 'data' in zoomed_channel_data[0]:
            first_col_len = len(zoomed_channel_data[0]['data'])
            #print(f"Actual first_col_len: {first_col_len}")
        else:
            first_col_len = 0
            #print(f"Actual overlap zoomed_channel_data: {first_col_len}")

        # Calculate the actual window length as a number
        actual_window_length = int((window_length /100) * first_col_len)             # Is that calculation right?

        # Calculate the actual overlap length as a number
        #actual_overlap_length = int((window_overlap /100) * actual_window_length)
        overlap_length_in_perc = window_overlap / 100               # Taking in percentage since Joao's SSM function taking the overlap in percentages

        features_results = []

        zoomed_channel_data = received_data.get('zoomedChannelData')    #converting the data in df
        df = pd.DataFrame({trace['name']: trace['data'] for trace in zoomed_channel_data})    # name= selected channel name, data= selected channel data
        #window_size = st.slider('window size', 10, len(s) // 3, len(s) // 10)

        # Initialize an empty list to store results
        features_results = []
        S = compute_ssm(df, actual_window_length, overlap_length_in_perc).tolist()


        # Calculate the maximum value for kernel size based on your formula
        max_kernel_size = int(len(df) / (overlap_length_in_perc - int(overlap_length_in_perc * 0.95))) # already defined .95

        # Set kernel_size to one-third of the maximum value, similar to your Streamlit default value
        kernel_size = max_kernel_size // 3

        # Ensure that kernel_size is at least 2
        #kernel_size = max(2, kernel_size)
        S_array = np.array(S) if isinstance(S, list) else S
        kernel_size = 10  # This should be an integer value representing half the kernel size
        nov_ssm = compute_novelty(S_array, kernel_size).tolist()

    # If it's not a POST request, maybe return a 400 Bad Request
    return JsonResponse({'ssm_data': S, 'nov_ssm': nov_ssm})


def feature_extraction(window_length, window_overlap_value, series):
    # Ensure window_length is an integer
    window_length = int(window_length)

    # Convert overlap percentage to a float fraction
    overlap = float(window_overlap_value) // 100

    # Calculate step based on window_length and overlap
    step = max(int(window_length - (window_length * overlap)), 1)

    # Initialize a list to store computed features
    features = []

    # Extract features for each window
    for i in range(0, len(series), step):
        window = series[i:i + window_length]

        if len(window) == window_length:
            window_features = {
                'mean': window.mean(),
                'standard_deviation': window.std(),
                'maximum': window.max(),
                'minimum': window.min()
            }
            features.append(window_features)

    return features
