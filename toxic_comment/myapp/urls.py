from django.urls import re_path
from . import views

urlpatterns = [
        re_path(r'^toxic_detect_post/$', views.toxic_detect_post, name='toxic_detect_post'),   # likepost view at /likepost
   ]