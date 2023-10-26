from django.urls import path
from . import views

urlpatterns = [
    path('preprocess/<int:file_id>/', views.preprocess_view, name='preprocess_view'),
    path('signal_visualization_view/<int:file_id>', views.signal_visualization_view, name='signal_visualization_view'),


]
