from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('fileUploadApp.urls')),  # Include the app's URL patterns

]