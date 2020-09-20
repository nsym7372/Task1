from firstapp.models import Animal
from django.shortcuts import render, redirect
from .forms import AnimalForm
# Create your views here.

def index(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == "POST":
        form = AnimalForm(request.POST)
        if form.is_valid():
            animal = Animal()
            animal.photo = request.FILES.get('photo')
            animal.save()
        return redirect('result')
    else:
        # form = BookForm()
    # return render(request, 'blog/new.html', {'form': form})
        return redirect('index')

def result(request):
    return render(request, 'result1.html')