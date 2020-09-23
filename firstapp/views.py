from firstapp.models import Animal
from django.shortcuts import render, redirect
from .forms import AnimalForm
from torchvision import transforms
from scipy.special import softmax
from .cnn_model import Net
from PIL import Image
import torch
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
        return redirect('index')

def result(request):
    data = Animal.objects.order_by('id').reverse()

    model = Net()
    model.load_state_dict(torch.load('firstapp/trained.pth', map_location=torch.device('cpu')))

    mt = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    img = Image.open(data[0].photo)

    model.eval()
    model.freeze()

    transformed = mt(img)
    ret1 = model(transformed.unsqueeze(0))
    ret = softmax(ret1)

    index = ret.argmax()
    rate = ret.squeeze().numpy()[index] * 100
    return render(request, 'result.html', {'photo':  data[0].photo, 'class': judgeClass(ret), 'rate': rate})

def judgeClass(ary):
    cls = ary.argmax()
    if cls == 0:
        return 'ごりら'
    elif cls == 1:
        return 'ぞう'
    elif cls == 2:
        return 'ぱんだ'
    else:
        return 'くまー'