from django.conf.urls import url,include
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    #url(r'^details/', views.details, name='details')
    url(r'^details/(?P<id>)', views.details, name='details'),
    url(r'accounts/', include('django.contrib.auth.urls'))
    #url(r'^details/(?P<id>\d+)/$', views.details, name='details')
]
