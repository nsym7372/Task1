from numpy.core.fromnumeric import argmax
from firstapp.models import Animal
from django.shortcuts import render, redirect
from .forms import AnimalForm
#from sklearn.externals import joblib
import joblib
from torchvision import transforms
from scipy.special import softmax
from .cnn_model import Net
import torch
# Create your views here.

# loaded_model = joblib.load('firstapp/trained_model.pkl')

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

# path = 'index.jpg'
# img2 = Image.open(path)
# img2

# mt = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ])

#     img3 = mt(img2)

#     ret = F.softmax(model(img3.unsqueeze(0)))
# print(ret)
# ret.argmax()
    model = Net()
    model.load_state_dict(torch.load('firstapp/trained.pth', map_location=torch.device('cpu')))

    mt = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    transformed = mt(data[0].photo)
    ret = softmax(model(transformed.unsqueeze(0)))
    



    # x = np.array([data[0]])
    # y = loaded_model.predict(x)
    # y_proba = loaded_model.predict_proba(x) * 100 # 100分率
    # y, y_proba = y[0], y_proba[0] # 配列からスカラーに変更

    # # 結果に応じた方のみ確率表示
    # proba = round(y_proba[y], 2)

    return render(request, 'result.html', {'photo':  data[0].photo})