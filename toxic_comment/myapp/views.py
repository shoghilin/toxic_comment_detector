from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .toxic_model.toxic_comment_detector import detect_toxic_comment

# Create your views here.
def sayhello(request):
    return HttpResponse("Hello Django!")

def BBS(request):
    return render(request, "BBS.html", locals())

def toxic_detect_post(request, *args, **kwargs):
    text = request.GET['text']
    result_bool, types = detect_toxic_comment(text)
    return JsonResponse({'toxic': bool(result_bool), 'toxic_type':types})