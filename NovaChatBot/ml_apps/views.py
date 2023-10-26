from fileUploadApp.models import UploadedFile
import xml.etree.ElementTree as ET
import json
from django.shortcuts import render, get_object_or_404
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def preprocess_view(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)

    file_content = None
    validation_errors = []

    if file.file:
        try:
            # Determine the file extension
            file_extension = file.file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(file.file)
                file_content = df.values.tolist()
            elif file_extension == 'txt':
                df = pd.read_csv(file.file, delimiter='\t')
                file_content = df.values.tolist()
            elif file_extension == 'xml':
                tree = ET.parse(file.file)
                root = tree.getroot()
                file_content = [[elem.text for elem in child] for child in root]
            elif file_extension == 'json':
                with open(file.file.path) as json_file:
                    data = json.load(json_file)
                    file_content = [[str(value) for value in row.values()] for row in data]
            elif file_extension == 'h5':
                df = pd.read_hdf(file.file)
                file_content = df.values.tolist()
            else:
                validation_errors.append("Unsupported file format.")

        except Exception as e:
            file_content = []
            validation_errors.append(f"Error occurred while reading the file: {str(e)}")

    else:
        validation_errors.append("File not found.")

    # Perform basic data validation checks
    if file_content:
        for row_index, row in enumerate(file_content):
            if any(pd.isnull(cell) for cell in row):
                validation_errors.append(f"Row {row_index + 1} contains null values.")

    return render(request, 'preprocess_view.html',
                  {'file_content': file_content, 'validation_errors': validation_errors, 'file_id': file.id})


def signal_visualization_view(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    # -----------> file id:::::  Joy hok nanar: ", file_id)
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
