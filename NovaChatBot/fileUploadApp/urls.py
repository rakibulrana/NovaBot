from django.urls import path
from . import views

urlpatterns = [
    path('', views.base, name='base'),
    path('upload/', views.upload_file, name='upload_file'),
    path('file_list_view/', views.file_list_view, name='file_list_view'),
    path('edit/<int:file_id>/', views.edit_file, name='edit_file'),
    path('delete/<int:file_id>/', views.delete_file, name='delete_file'),
    path('sign_in/', views.sign_in, name='sign_in'),
    path('save_user_signin_data', views.save_user_signin_data, name='save_user_signin_data')

]
