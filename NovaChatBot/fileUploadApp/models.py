# from django.db import models
#
#
# class UploadedFile(models.Model):
#     name = models.CharField(max_length=255)
#     file = models.FileField(upload_to='uploaded_files/')
#     uploaded_date = models.DateTimeField(auto_now_add=True)
#
#     def __str__(self):
#         return self.name


from django.db import models


class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    original_name = models.CharField(max_length=255, default="No original name")
    uploaded_date = models.DateTimeField(auto_now_add=True)

    # name = models.CharField(max_length=100, default='O')
    # age = models.IntegerField(default='0')
    # GENDER_CHOICES = [
    #     ('M', 'Male'),
    #     ('F', 'Female'),
    #     ('O', 'Other'),
    # ]
    # gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='O')

    # Add more fields as needed

    def __str__(self):
        return self.name  # This is the string representation of the model
