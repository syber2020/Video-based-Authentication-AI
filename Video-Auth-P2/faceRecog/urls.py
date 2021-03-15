
from django.conf.urls import url, include
from django.contrib import admin
from faceRecog import views as app_views
from django.contrib.auth import views

urlpatterns = [
    url(r'^$', app_views.index),
    url(r'^error_image$', app_views.errorImg),
    url(r'^trainer$', app_views.trainer),
    url(r'^detect$', app_views.detect),
    url(r'^capture$', app_views.capture),
    url(r'^admin/', admin.site.urls),
    url(r'^records/', include('records.urls')),
]
