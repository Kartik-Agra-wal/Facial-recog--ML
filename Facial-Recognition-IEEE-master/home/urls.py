from django.urls import path
from home.views import Index, webcam_feed

urlpatterns = [
    path('', Index, name='index'),
    path('webcam_feed/', webcam_feed, name='webcam_feed'),
]