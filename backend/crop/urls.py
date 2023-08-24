from django.urls import path
from . import views 

urlpatterns = [
        path('home/form-data/', views.FormDataView.as_view(), name="form-data"),
        path('aboutus/', views.aboutus, name="about-us"),
        ]
