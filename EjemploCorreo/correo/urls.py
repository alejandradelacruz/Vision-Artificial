from django.urls import path
from . import views

urlpatterns = [
    path("mail/", views.correo, name="mail"),
    path("", views.index, name='index'),
]


