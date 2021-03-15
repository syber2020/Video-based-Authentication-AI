# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import Records
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


def index(request):
    records = Records.objects.all()[:10]
    context = {
        'records': records
    }
    return render(request, 'records.html', context)


def details(request, id):
    print("ID being queried:" + id)
    request_str = str(request)
    #request_str.split("WSGIRequest: GET '/records/details/"))[1].split("'>")[0]
    res = request_str.partition("details/")[2].partition("'>")[0]
    print("res:" + res)
    record = Records.objects.get(id=res)
    print(record)
    context = {
        'record': record
    }
    return render(request, 'details.html', context)
