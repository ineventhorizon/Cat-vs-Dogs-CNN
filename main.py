import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    #print(device)
    print(f'Running on {torch.cuda.get_device_name(device)}')
    print(f'CUDA Device count: {torch.cuda.device_count()}')
else:
    device = torch.device('cpu')
    print('Running on CPU')


REBUILD_DATA = False #True, If you changed something in the data
REBUILD_TEST_DATA = True #True, If you added new images to testing folder
TRAIN_EXISTING_MODEL = True #If there is already a mytraining.pt in the same path True, If you want to train a new model then False
TRAIN_DATA = True #True if you want to train the model otherwise False



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 Input, 32 Convolutional Feature, 5 Kernel Size
        self.conv2 = nn.Conv2d(32, 64, 5)  # 32 Input, 64 Convolutional Feature, 5 Kernel Size
        self.conv3 = nn.Conv2d(64, 128, 5)  # 64 Input, 128 Convolutional Feature, 5 Kernel Size
        self.fc1 = nn.Linear(128*2*2, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        #First Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        #Second Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        #Third Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #print(torch.flatten(x, 1).shape)
        #To pass x to fully connected layer we need to apply flatten to x
        x = torch.flatten(x, 1)
        #First fully connected layer, Fully connected layer -> relu activation function
        x = self.fc1(x)
        x = F.relu(x)
        # Second fully connected layer, Fully connected layer -> sigmoid activation (output)
        x = self.fc2(x)
        x = F.softmax(x, 1)
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        #For finding fc1 input size
        #print(x.shape)
        #print(x[0].shape)
        return x

#For making Dogs vs Cats data
class DogsVSCats():
    IMG_SIZE = 50
    CATS = 'CatnDogs/Cat'
    DOGS = 'CatnDogs/Dog'
    MY_TEST = "CatnDogs/MyTest"
    LABELS = {CATS: 0, DOGS: 1}
    CLASS_COUNT = 2
    training_data = []
    my_testdata = []
    catcount = 0
    dogcount = 0


    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    #img = cv2.Laplacian(img, cv2.CV_64F)
                    # 1,0 Cat
                    # 0,1 Dog
                    self.training_data.append([np.array(img), np.eye(self.CLASS_COUNT)[self.LABELS[label]]])
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    #print(str(e))
                    pass

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print(f'Cats : {self.catcount}, Dogs : {self.dogcount}')

    def make_test_data(self):
        for f in tqdm(os.listdir(self.MY_TEST)):
            try:
                path = os.path.join(self.MY_TEST, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                self.my_testdata.append((np.array(img), f))
            except Exception as e:
                pass
        np.save('my_testdata.npy', self.my_testdata)

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            optimizer.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            #plt.plot(epoch, loss.detach().numpy())
            #plt.show()

        print(f'Epoch {epoch} loss: {loss}')
        #torch.save(net.state_dict(), 'mytraining.pt')

def validate(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
            #print(net_out)
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print(f'Accuracy {round(correct/total, 3)}')
    return correct/total


if REBUILD_DATA:
    dogvcats = DogsVSCats()
    dogvcats.make_training_data()
if REBUILD_TEST_DATA:
    dogvcats = DogsVSCats()
    dogvcats.make_test_data()

training_data = np.load('training_data.npy', allow_pickle=True)
mytest_data = np.load('my_testdata.npy', allow_pickle=True)

if TRAIN_DATA:
    BATCH_SIZE = 100#The number of training examples in one forward/backward pass
    EPOCHS = 10 #Number of forward pass and backward pass of all the training examples
    VAL_PCT = 0.10 #Percentage of validation data out of training data
    #Traindatasize/Batch_size iterations are required to complete 1 Epoch


    #Train tensors
    X = torch.tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data])

    val_size = int(len(X)*VAL_PCT)
    print(val_size, 'Val size')

    #Train data
    train_X = X[:-val_size]
    train_y = y[:-val_size]

    #Validation data
    test_X = X[-val_size:]
    test_y = y[-val_size:]
    print(len(train_X), len(test_X), 'ValidationX, TestX')

    net = Net().to(device)
    print(f'Type of conv1 weights: {net.conv1.weight.type()}')
    print(f'Type of conv2 weights: {net.conv2.weight.type()}')
    print(f'Type of conv3 weights: {net.conv3.weight.type()}')
    print(f'Type of fc1 weights: {net.fc1.weight.type()}')
    print(f'Type of fc2 weights: {net.fc2.weight.type()}')
    if TRAIN_EXISTING_MODEL:
        net.load_state_dict(torch.load('mytraining.pt'))
        net.eval()

    print(net)

    #Train model
    train(net)
    #Test model
    validate(net)

    torch.save(net.state_dict(), 'mytraining.pt')
elif not TRAIN_DATA:
    X_Test = torch.tensor([i[0] for i in mytest_data]).view(-1, 50, 50).to(device)
    X_Test = X_Test / 255.0
    y_Test = [i[1] for i in mytest_data]

    net = Net().to(device)
    net.load_state_dict(torch.load('mytraining.pt'))
    net.eval()
    print('\n')

    # Argmax 0 = 1,0 Cat
    # Argmax 1 = 0,1 Dog
    with torch.no_grad():
        for i in tqdm(range(len(X_Test))):
            out = net(X_Test[i].view(-1, 1, 50, 50))[0]
            oct = net(X_Test[i].view(-1, 1, 50, 50))
            predicted_class = torch.argmax(out)
            predicted_class1 = torch.argmax(oct)
            #print(predicted_class, "predicted [0]")
            #print(predicted_class1, "predicteddfdd")
           # print(out, "class [0]")
            if predicted_class == 1:
                print(f'{y_Test[i]} is a Dog  {out}')
            elif predicted_class == 0:
                print(f'{y_Test[i]} is a Cat {out} ')
