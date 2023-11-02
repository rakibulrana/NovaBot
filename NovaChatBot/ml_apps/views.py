from fileUploadApp.models import UploadedFile

import matplotlib.pyplot as plt
from io import BytesIO
import base64

import pandas as pd
from django.shortcuts import render, get_object_or_404
from ydata_profiling import ProfileReport  # Import ProfileReport from ydata_profiling
import plotly.graph_objs as go
import json

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

            selected_channels = [int(channel) - 1 for channel in selected_channels]
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
