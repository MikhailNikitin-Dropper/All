import numpy as np
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim, torch
from typing import Final
from torch import nn

TRAIN_BATCH_SIZE: Final[int] = 64
REAL_BATCH_SIZE: Final[int] = 64
SIZE_OF_PIC_SIDE: Final[int] = 28
SIZE_OF_PIC: Final[int]=SIZE_OF_PIC_SIDE*SIZE_OF_PIC_SIDE

EPOCH: Final[int] = 10 #количество эпох

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]) #переменная для форматирования данных
# достаём данные из MINST для обучения и тестирования
trainset = datasets.MNIST('TRAINSET', download=True, train=True, transform=transform)
realset = datasets.MNIST('TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True) #делим данные на батчи и перемешиванием
realloader = torch.utils.data.DataLoader(realset, batch_size=REAL_BATCH_SIZE, shuffle=True)    #делим данные на батчи и перемешиванием
dataiter = iter(trainloader)
images, nums = dataiter.__next__() #images - изображения, nums - соответтвующие им числа
#print(images[0])
#print(nums[1::])
#Вывод одного батча, на котором обучается нейросеть
for index in range(TRAIN_BATCH_SIZE):
    plt.subplot(8, 8, index+1)
    plt.imshow(images[index].resize_(1, SIZE_OF_PIC_SIDE, SIZE_OF_PIC_SIDE).numpy().squeeze(), cmap='gray_r')
plt.show()

#Иницализируем константы для размера слоёв нейроной сети - входной и два внутренних
input_size = SIZE_OF_PIC
hidden_size1 = 128
hidden_size2 = 64
output_size = 10

#Создаём секвенционную (последовательную сеть) сеть
model = nn.Sequential(nn.Linear(input_size, hidden_size1),
                      nn.ReLU(),
                      nn.Linear(hidden_size1, hidden_size2),
                      nn.ReLU(),
                      nn.Linear(hidden_size2, output_size),
                      nn.LogSoftmax(dim=1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #подключение cuda
print(device)
model.to(device)

criterion = nn.NLLLoss() #для подсчёта отклонения после прохода через сеть
images = images.view(images.shape[0], -1) #превращаем обучающие картинки в вектора
logps = model(images.cuda())  #вводим наши обучающие картинки
loss = criterion(logps, nums.cuda()) #передаём модель с картинками и значения в "подсчётчик" отклонения
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9) #Реализует стохастический градиентный спуск c импульсом 0.9
images.resize_(TRAIN_BATCH_SIZE, SIZE_OF_PIC_SIDE * SIZE_OF_PIC_SIDE) #изменяем размер картинок
optimizer.zero_grad() #очищаем градиент

time0 = time() #время начала работы

for epoch in range(EPOCH):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1) #превращаем обучающие картинки в вектора

        optimizer.zero_grad()
        output = model(images.cuda())
        loss = criterion(output, labels.cuda())
        loss.backward() #обучение прохождением нашего вектора в обратном порядке
        optimizer.step() #оптимизация весов
        running_loss += loss.item() #считаем потери
    else:
        print("Epoch {} - Training loss: {} \n".format(epoch, running_loss / len(trainloader)))

print("Training time= {} ".format(time() - time0))

def view_results(img, ps): #выводит изображение и вероятности значений
    ps = ps.cpu().data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, SIZE_OF_PIC_SIDE, SIZE_OF_PIC_SIDE).numpy().squeeze(), cmap='gray_r')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

images, labels = iter(realloader).__next__() #берём реальную базу данных

img = images[0].view(1, SIZE_OF_PIC)
with torch.no_grad():
    logps = model(img.cuda())
ps = torch.exp(logps) #т.к. вывод вероятностей у нас в виде логарифмов вероятностей - берём экспоненту
probability = list(ps.cpu().numpy()[0])
print("Predicted Digit = {} \n".format(probability.index(max(probability)))) #выводим наиболее вероятную цифру на 0ой картинке
view_results(img.view(1, SIZE_OF_PIC_SIDE, SIZE_OF_PIC_SIDE), ps) #выводим нулевую картинку
plt.show()

correct_count, all_count = 0, 0
for images,labels in realloader:
  for i in range(len(labels)):
    img = images[i].view(1, SIZE_OF_PIC)

    with torch.no_grad():
        logps = model(img.cuda())

    ps = torch.exp(logps)
    probability = list(ps.cpu().numpy()[0])
    pred_num = probability.index(max(probability))
    true_num = labels.numpy()[i]
    if(true_num == pred_num):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested = {} \n".format(all_count))
print("Model Accuracy = {} \n".format(correct_count/all_count))