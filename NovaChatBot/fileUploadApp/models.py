from django.db import models


class UploadedFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploaded_files/')
    uploaded_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name