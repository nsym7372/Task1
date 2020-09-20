from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('result', views.result, name='result'),
    # path('upload', views.upload.as_view(), name='upload'),
    
]