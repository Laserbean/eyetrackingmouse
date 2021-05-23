import numpy as np
# import eyetrack as eye
import cv2
import os
import copy

import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import matplotlib.pyplot as plt

'''
TODO: add sigmoid function 



'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
class mynn(torch.nn.Module):
    def __init__(self):
        super(mynn, self).__init__()
        depth = 1
        self.layer = nn.Sequential(
            nn.Conv2d(1, depth, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(depth)#,
            ###nn.Flatten(),            
            ###nn.Linear(40*40, 1)
            )
        
        ###f2 = 1 #number of output channels
        ###self.layer = nn.Sequential(
            ###nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            ###nn.ReLU(),
            ###nn.BatchNorm2d(f2),
            ###nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(36 * 36 * depth, 160) #fully connected layer
        #self.fc2 = nn.Sequential(
            #nn.ReLU(),
            #nn.Linear(160, 20)
            #)
        self.fc2 = nn.Linear(160, 20)
     
        self.fc3 = nn.Linear(20, 2)
        #self.fc3 = nn.Sequential(
            #nn.Flatten(),
            #nn.Linear(20, 2)
            #)

    def forward(self,x):
        batch_size = 1
        #print(x.size())
        #x = x.view(batch_size, -1)
        
        ## use ^^^ to get the number 1600
        x = self.layer(x)
        #print(x.size())
        ###x = self.layer(x)
        ###x = x.reshape(x.size(0), -1)
        x = x.view(1, -1)
        #print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def dataLoad(paths, want = 0):
    #print(nameList)

    totalHolder = []
    #dims = [1440,900]
    w1 = 1920
    h1 = 1080  
    w = w1
    h = h1
    if (want == 0):
        w = 1
        h = 1 
    for path in paths:
        nameList = os.listdir(path)
        try:
            nameList.remove(".DS_Store")
        except:
            pass        
        for name in nameList:
            im = cv2.cvtColor(cv2.imread(path + "/" + name), cv2.COLOR_BGR2GRAY)
            nnim = torch.tensor([[im]]).to(dtype=torch.float,device=device)
            #print(nnim.size())
            sname = name.split(".")
            coord = [float(sname[0])/w - w1/2*want , float(sname[1])/h  -h1/2*want]
            #print(coord)
            nncoord = torch.tensor(coord).to(dtype=torch.float,device=device)
            totalHolder.append([nnim, nncoord])
            #top = max([max(x) for x in im])
            #totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                                #torch.tensor([[int((name.split("."))[want])/dims[want]]])
                                #.to(dtype=torch.float,device=device)))            
    #print(totalHolder)              
    return totalHolder


def evaluateModel(model,testSet, sidelen = 1920):
    model.eval()
    err = 0
    i = 1
    for (im, label) in testSet:
        i = i+1
        output = model(im)
        outx = output[0][0].item()
        outy = output[0][1].item()     
        out = (outx, outy)

        labx = label[0].item()
        laby = label[1].item()
        lab = (labx, laby)
        
        #err += abs(output.item() - label.item())
        err += ((laby-outy)**2 + (labx-outx)**2)**0.5
    model.train()

    return (err/i)

num_epochs = 11

bigTest = []
bigTrain = []

def testTrainNN(trainingSet, testSet):
    model = mynn().to(device)
    model.train()
    np.random.shuffle(trainingSet)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []    
    for epoch in range(num_epochs):
        print("epoch", epoch)
        np.random.shuffle(trainingSet)        
        for i,(im, label) in enumerate(trainingSet):
            #print(label[0])
            output = model(im)
            #print(im.size())
            #print("output", output)
            loss = criterion(output, label.unsqueeze(0))
            #print("label", label)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            if (i+1) % 500 == 0:
                print(i+1)        
                testSc = evaluateModel(model,testdata)
                trainSc = evaluateModel(model,totaldata)
                if testSc < bestScore:
                    bestModel = copy.deepcopy(model)
                    bestScore = testSc
                testscores.append(testSc)
                trainscores.append(trainSc)      

                print("training score = ", trainSc)
                print("test score = ", testSc)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, len(trainingSet), loss.item()))                
    
    bigTest.append(testscores)
    bigTrain.append(trainscores)

    finalScore = evaluateModel(bestModel,testdata)
    # finalScore = evaluateModel(bestModel,test,sidelen=900)
    print("final score = ", finalScore)    

    #if finalScore < 150:
        #torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore))+".plt")    
    torch.save(bestModel.state_dict(), "modeltest.pth")
    print("Saved PyTorch Model State to modeltest.pth")    
    

def trainModel():
    model = ConvNet().to(device)
    # model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ###bestModel = model
    ###bestScore = 10000
    ###testscores = []
    ###trainscores = []

    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        np.random.shuffle(trainingSet)

        for i,(im, label) in enumerate(trainingSet):

            output = model(im)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print(i+1)
                # testSc = evaluateModel(model,test,sidelen=900)
                testSc = evaluateModel(model,test)
                # trainSc = evaluateModel(model,trainingSet,sidelen=900)
                trainSc = evaluateModel(model,trainingSet)
                if testSc < bestScore:
                    bestModel = copy.deepcopy(model)
                    bestScore = testSc
                testscores.append(testSc)
                trainscores.append(trainSc)

                print(trainSc)
                print(testSc)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, len(trainingSet), loss.item()))

    bigTest.append(testscores)
    bigTrain.append(trainscores)

    finalScore = evaluateModel(bestModel,test)
    # finalScore = evaluateModel(bestModel,test,sidelen=900)
    print(finalScore)    

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore))+".plt")

    # plt.title(str(int(finalScore)))
    # plt.plot(testscores)
    # plt.plot(trainscores)

    
    

import pickle

loaddata = True;
#loaddata = False;
if (loaddata):
    print("saving or didn't work?")
    file = open("mytorch.obj", "wb")
    totaldata = dataLoad(["mydata", "mydata2", "mydata3"])
    
    pickle.dump(totaldata, file)
else:
    file = open("mytorch.obj", "rb")
    totaldata = pickle.load(file)
    print("loading")
    
testdata = dataLoad(["mytest", "mytest2"])
print(totaldata[0])
testTrainNN(totaldata, testdata)