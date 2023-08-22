from django.urls import path
from . import views

urlpatterns = [
    path('preprocess/<int:file_id>/', views.preprocess_view, name='preprocess_view'),
]