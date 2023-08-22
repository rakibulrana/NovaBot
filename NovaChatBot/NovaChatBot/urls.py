from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('fileUploadApp.urls')),  # Include the app's URL patterns
    path('ml_apps/', include('ml_apps.urls')),  # Add this line for the ml_apps app


]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)