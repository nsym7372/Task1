from django.shortcuts import render, redirect

# Create your views here.

def index(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == "POST":
        # form = BookForm(request.POST)
        # if form.is_valid():
        #     book = Book()
        #     print(request)
        #     book.title = request.POST['title']
        #     book.link = request.POST['link']
        #     book.image = request.FILES['image']
        #     book.author = request.user
        #     book.published_date = timezone.now()
        #     book.save()
        return redirect('result')
    else:
        # form = BookForm()
    # return render(request, 'blog/new.html', {'form': form})
        return redirect('index')

def result(request):
    return render(request, 'result.html')