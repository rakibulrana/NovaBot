from fileUploadApp.models import UploadedFile
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from django.shortcuts import render, get_object_or_404
from ydata_profiling import ProfileReport  # Import ProfileReport from ydata_profiling
from django.shortcuts import render
import tsfel  # Import TSFEL or your feature extraction library
from django.http import JsonResponse  # Import JsonResponse for AJAX responses

profile_config = {
    "html.navbar_show": False,
}


def preprocess_view(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    file_content = None
    validation_errors = []
    profile_html = None
    num_columns = 0  # Initialize the number of columns
    column_names = []  # Initialize column names list

    if file.file:
        try:
            # Determine the file extension
            file_extension = file.file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(file.file)
            elif file_extension == 'txt':
                df = pd.read_csv(file.file, delimiter='\t')
            elif file_extension == 'xml':
                # Your XML processing code here
                pass
            elif file_extension == 'json':
                # Your JSON processing code here
                pass
            elif file_extension == 'h5':
                # Your HDF processing code here
                pass
            else:
                validation_errors.append("Unsupported file format")

            # Check for NaN values in the DataFrame and fill them if necessary
            if df.isnull().values.any():
                df.fillna(0, inplace=True)  # You can fill NaN values with any appropriate value

            # Dynamically generate column names based on the number of columns
            num_columns = len(df.columns)
            column_names = [f'channel{i}' for i in range(1, num_columns + 1)]
            df.columns = column_names

            df_head = df.head()
            file_content = df.values.tolist()

        except Exception as e:
            validation_errors.append(f"Error occurred while reading the file: {str(e)}")

    else:
        validation_errors.append("File not found.")

    selected_channels = []  # Initialize the selected channels variable
    if request.method == "POST" and "generate_report" in request.POST:
        # Generate the Pandas Profiling Report
        if file_content:
            # Create a DataFrame from your file_content
            df = pd.DataFrame(file_content, columns=column_names)
            selected_channels = df.columns  # Get the channel names from DataFrame

            title = "Pandas Profiling Report"

            # Create the profile report
            profile = ProfileReport(df)

            # Convert the report to an HTML string
            profile_html = profile.to_html()

    return render(request, 'preprocess_view.html', {
        'file_content': file_content,
        'validation_errors': validation_errors,
        'file_id': file.id,
        'df_head': df_head,
        'profile_html': profile_html,
        'column_names': column_names,  # Pass the dynamically generated column names to the template
        'num_columns': num_columns,  # Pass the number of columns to the template
        'selected_channels': selected_channels  # Pass selected channel names to the template
    })


def visualize_channels(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)

    selected_channels = request.GET.get('channels').split(',')
    print(selected_channels, " Print my selected_:channels")
    print(" My file id: ", file_id)
    reconstructed_data = {}  # A dictionary to store the reconstructed data
    # dk = pd.DataFrame(file)
    # print(dk)
    if file.file:
        try:

            # Determine the file extension
            file_extension = file.file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(file.file)
            elif file_extension == 'txt':
                df = pd.read_csv(file.file, delimiter='\t')
            # Add more file format handling here as needed
            else:
                # Handle unsupported file formats
                return render(request, 'error.html', {'error_message': 'Unsupported file format'})

            selected_channels = [int(channel) - 1 for channel in
                                 selected_channels]  # usually got selected from the next one! therefore, we needed to correct.
            for col_idx, column_name in enumerate(df.columns):
                if col_idx in selected_channels:
                    channel_data = df[column_name].tolist()
                    reconstructed_data[column_name] = channel_data
            # for channel in selected_channels:
            #     if channel in df.columns:
            #         for i in 
            #         # Retrieve data for the selected channel as a list
            #         channel_data = df[channel].tolist()
            #         reconstructed_data[channel] = channel_data

            return render(request, 'visualize_channels.html', {
                'selected_channels': selected_channels,
                'reconstructed_data': reconstructed_data,
            })

        except Exception as e:
            # Handle exceptions related to file processing
            return render(request, 'error.html',
                          {'error_message': f"Error occurred while processing the file: {str(e)}"})

    else:
        # Handle the case where the file is not found
        return render(request, 'error.html', {'error_message': 'File not found'})


def view_features_calculation(request, file_id):
    # file = get_object_or_404(UploadedFile, id=file_id)
    #
    # selected_channels = request.GET.get('channels').split(',')
    #
    # selected_features = request.GET.get('features').split(',')
    #
    # window_overlap_value = request.GET.get('windowOverLapValue').split(',')
    # window_length = request.GET.get('windowLength').split(',')
    file = get_object_or_404(UploadedFile, id=file_id)

    selected_channels = [3, 4]
    selected_features = [2, 3, 4]
    window_overlap_value = 50
    window_length = 12

    if file.file:
        try:

            # Determine the file extension
            file_extension = file.file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(file.file)
            elif file_extension == 'txt':
                df = pd.read_csv(file.file, delimiter='\t')
            # Add more file format handling here as needed
            else:
                # Handle unsupported file formats
                return render(request, 'error.html', {'error_message': 'Unsupported file format'})

            # Calculate mean for each selected channel
            features_results = []

            # usually got selected from the next one! therefore, we needed to correct.
            selected_channels = [int(channel) - 1 for channel in selected_channels]
            for col_idx, column_name in enumerate(df.columns):
                if col_idx in selected_channels:
                    channel_data = df[column_name].tolist()
                    channel_features = feature_extraction(window_length, window_overlap_value, channel_data)
                    features_results.append(channel_features)

            transposed_stats = {
                'mean': [],
                'standard_deviation': [],
                'maximum': [],
                'minimum': []
            }

            # Populate the transposed_stats structure
            for channel in features_results:
                for stats in channel:
                    transposed_stats['mean'].append(stats['mean'])
                    transposed_stats['standard_deviation'].append(stats['standard_deviation'])
                    transposed_stats['maximum'].append(stats['maximum'])
                    transposed_stats['minimum'].append(stats['minimum'])

            # Pass both features_results and transposed_stats to the template
            return render(request, 'view_features_calculation.html', {
                'features_results': features_results,
                'transposed_stats': transposed_stats
            })
        except Exception as e:
            # Handle exceptions related to file processing
            return render(request, 'error.html',
                          {'error_message': f"Error occurred while processing the file: {str(e)}"})

    else:
        # Handle the case where the file is not found
        return render(request, 'error.html', {'error_message': 'File not found'})


# Feature extraction window length, overlap selection function


def feature_extraction(window_length, window_overlap_value, channel_data):
    dataset = channel_data
    window_size = window_length
    overlap = window_overlap_value

    step = max(int(window_size - window_size * overlap), 1)

    # Initialize a list to store computed features
    features = []

    # Extract features for each window
    for i in range(0, len(dataset), step):
        window = dataset[i:i + window_size]

        if len(window) == window_size:
            window_features = {
                'mean': np.mean(window),
                'standard_deviation': np.std(window),
                'maximum': np.max(window),
                'minimum': np.min(window)
            }
            features.append(window_features)

    return features


def signal_visualization_view(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)

    file_content = None
    validation_errors = []

    if file.file:
        try:
            df = pd.read_csv(file.file, header=None, delimiter='\t')
            signals = df.values
            num_samples = signals.shape[0]

            # Reconstruct and plot signals
            plt.figure(figsize=(10, 6))
            for i in range(1, signals.shape[1]):
                plt.plot(range(num_samples), signals[:, i], label=f"Signal {i}")

            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.title("Reconstructed Signals")
            plt.legend()

            # Convert the plot to an image for display on the web page
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            validation_errors.append(f"Error occurred while processing the dataset: {str(e)}")
            image_base64 = None
    else:
        validation_errors.append("File not found.")
        image_base64 = None

    return render(request, 'signal_visualization.html',
                  {'validation_errors': validation_errors, 'image_base64': image_base64, 'file_id': file.id})
