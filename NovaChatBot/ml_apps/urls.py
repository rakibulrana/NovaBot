from django.urls import path
from . import views

urlpatterns = [
    path('preprocess/<int:file_id>/', views.preprocess_view, name='preprocess_view'),
    path('signal_visualization_view/<int:file_id>', views.signal_visualization_view, name='signal_visualization_view'),
    path('visualize_channels/<int:file_id>/', views.visualize_channels, name='visualize_channels'),
    path('view_features_calculation/<int:file_id>/', views.view_features_calculation, name='view_features_calculation'),

]
