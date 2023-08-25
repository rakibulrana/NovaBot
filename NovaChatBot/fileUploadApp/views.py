from datetime import timezone
from django.shortcuts import render, redirect, get_object_or_404
from .models import UploadedFile
from .forms import UploadFileForm
from django.contrib import messages
import os
from django.http import FileResponse  # viewing and reading files
import csv


def base(request):
    return render(request, 'base.html', {'year': 2023})


# for git problem fixing comment
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['file']
            file_name = file.name
            new_entry = UploadedFile(file=file, original_name=file_name)

            new_entry.save()
            messages.success(request, "File uploaded successfully!")
            return redirect('file_list_view')

            # Redirect to a success page or perform other actions
    else:
        form = UploadFileForm()
    return render(request, 'fileUpload/upload_file.html', {'form': form})


# function for view data content, csv turns like working for all files!
def display_csv_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)

    if file.file:
        try:
            decoded_file = file.file.read().decode('utf-8')
            csv_data = csv.reader(decoded_file.splitlines())
            file_content = "\n".join(",".join(row) for row in csv_data)
        except Exception as e:
            file_content = f"Error occurred while reading the CSV file: {str(e)}"
    else:
        file_content = "File not found."

    return render(request, 'fileUpload/display_file.html', {'file_content': file_content})


def file_list_view(request):
    files = UploadedFile.objects.all()
    return render(request, 'fileUpload/file_list_view.html', {'files': files})


def edit_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    if request.method == 'POST':
        form = UploadFileForm(request.POST, instance=file)
        if form.is_valid():
            # Check if 'file' field has changed
            if 'file' in form.changed_data:
                # Update the file instance with the new file data
                file = form.save(commit=False)
                file.uploaded_date = timezone.now()  # Update the uploaded date
                file.save()
                messages.success(request, 'File name and file updated successfully.')
            else:
                # Only update the file name
                file.original_name = form.cleaned_data['original_name']
                file.save()
                messages.success(request, 'File name updated successfully.')

            return redirect('file_list_view')
    else:
        form = UploadFileForm(instance=file)
    return render(request, 'fileUpload/edit_file.html', {'form': form, 'file': file})


def delete_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    if request.method == 'POST':
        file.delete()
        return redirect('file_list_view')
    return render(request, 'fileUpload/delete_file.html', {'file': file})


def download_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    file_path = file.file.path
    filename = os.path.basename(file_path)
    response = FileResponse(open(file_path, 'rb'))
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def sign_in(request):
    return render(request, 'loginController/signin.html')


def save_user_signin_data(request):
    return render(request, 'fileUpload/upload_file.html')
