from torch import argmax
from firstapp.models import Animal
from django.shortcuts import render, redirect
from .forms import AnimalForm
from torchvision import transforms
from scipy.special import softmax
# from .ml_model.cnn_model import Net
from PIL import Image
import onnxruntime
import torch
import numpy as np
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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def result(request):

    # 保存したデータを学習用に調整
    data = Animal.objects.order_by('id').reverse()

    mt = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    img = Image.open(data[0].photo)
    transformed = mt(img)

    # モデル復元
    ort_session = onnxruntime.InferenceSession("firstapp/ml_model/model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(transformed.unsqueeze(0))}
    ret = ort_session.run(None, ort_inputs)

    # ret1 = model(transformed.unsqueeze(0))
    # ret = softmax(ret1)
    
    
    index = np.argmax(ret)
    # rate = ret.squeeze().numpy()[index] * 100
    rate = softmax(ret[0][0])[index] * 100
    return render(request, 'result.html', {'photo':  data[0].photo, 'class': judgeClass(index), 'rate': rate})
    # return render(request, 'result.html')

# def result(request):
#     data = Animal.objects.order_by('id').reverse()

#     model = Net()
#     model.load_state_dict(torch.load('firstapp/ml_model/trained.pth', map_location=torch.device('cpu')))

#     mt = transforms.Compose([
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#         ])
    
#     img = Image.open(data[0].photo)

#     model.eval()
#     model.freeze()

#     transformed = mt(img)
#     ret1 = model(transformed.unsqueeze(0))
#     ret = softmax(ret1)

#     index = ret.argmax()
#     rate = ret.squeeze().numpy()[index] * 100
#     return render(request, 'result.html', {'photo':  data[0].photo, 'class': judgeClass(ret), 'rate': rate})

def judgeClass(idx):

    if idx == 0:
        return 'ごりら'
    elif idx == 1:
        return 'ぞう'
    elif idx == 2:
        return 'ぱんだ'
    else:
        return 'くまー'