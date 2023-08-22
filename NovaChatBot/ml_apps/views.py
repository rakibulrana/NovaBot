from django.shortcuts import render, redirect, get_object_or_404
from fileUploadApp.models import UploadedFile
from .utils import preprocess_csv


def preprocess_view(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)

    if request.method == 'POST':
        # Assuming you have a preprocess_csv function that performs data preprocessing
        result = preprocess_csv(file.file.path)
        # You can save the result in the database or use it for further processing

    return render(request, 'preprocess_view.html', {'file': file})


