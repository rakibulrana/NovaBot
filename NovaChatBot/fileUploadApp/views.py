from django.shortcuts import render, redirect, get_object_or_404
from .models import UploadedFile
from .forms import UploadFileForm


def base(request):
    return render(request, 'base.html', {'year': 2023})


def upload_file(request):
    if request.method == 'POST':

        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            return redirect('file_list_view')
    else:
        form = UploadFileForm()

    return render(request, 'fileUpload/upload_file.html', {'form': form})


def file_list_view(request):
    files = UploadedFile.objects.all()
    return render(request, 'fileUpload/file_list_view.html', {'files': files})


def edit_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    if request.method == 'POST':
        form = UploadFileForm(request.POST, instance=file)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = UploadFileForm(instance=file)
    return render(request, 'fileUpload/edit_file.html', {'form': form, 'file': file})


def delete_file(request, file_id):
    file = get_object_or_404(UploadedFile, id=file_id)
    if request.method == 'POST':
        file.delete()
        return redirect('home')
    return render(request, 'fileUpload/delete_file.html', {'file': file})


def sign_in(request):

    return render(request, 'loginController/signin.html')


def save_user_signin_data(request):

    return render(request, 'fileUpload/upload_file.html')