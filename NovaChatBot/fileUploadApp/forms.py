from django import forms
from .models import UploadedFile


class UploadFileForm(forms.ModelForm):
    name = forms.CharField(label='Name', max_length=100)
    age = forms.IntegerField(label='Age')
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    gender = forms.ChoiceField(label='Gender', choices=GENDER_CHOICES)

    file = forms.FileField(label='File Upload')

    class Meta:
        model = UploadedFile
        fields = ['name', 'age', 'gender', 'file']

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            allowed_formats = ['csv', 'txt', 'xml', 'json', 'h5']
            ext = file.name.split('.')[-1].lower()
            if ext not in allowed_formats:
                raise forms.ValidationError(f"Invalid file format. Allowed formats are: {', '.join(allowed_formats)}")
        return file
