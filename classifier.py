
# file containing definitions for Classifier

import torch.nn as nn
import torch
# import config_local
import config
import os

class Deep_Classifier(nn.Module):
    def __init__(self,num_classes):
        super(Deep_Classifier, self).__init__()


        # self.bn1 = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=128,kernel_size=3)

        self.conv2 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3)

        self.point_wise = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=(1,1))


        self.max_pool = nn.MaxPool2d(kernel_size=(2,2))

        self.fc1 = nn.Linear(6144,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,num_classes)

        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
    
    def embed(self,x,last=1):
        
        x =torch.unsqueeze(x, 1) # adding a channel dimension
        x = x.permute(0,1,3,2)


        # x = self.bn1(x)
        

        out = self.conv1(x)
        out  = self.max_pool(out)
        out = self.relu(out)

        out = self.conv2(out)
        out  = self.max_pool(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out  = self.max_pool(out)
        out = self.relu(out)
        out = self.point_wise(out)
        # out = out.view(out.shape[0],-1)

        # print("out.shape -> ",out.shape)
        # return
        
        # flattening the output
        out = out.view(out.shape[0],-1)

        out = self.dropout(out)

        out = self.relu(self.fc1(out))
        # out = self.relu(self.fc2(out))
        if last == 1:
            return out 
        
        out = self.relu(self.fc3(out))
        
        if last == 2:
            return out 

        out = self.relu(self.fc4(out))
        
        if last == 3:
            return out 
        
        out = self.relu(self.fc5(out))
        
        return out


    def forward(self, x):
        
        x =torch.unsqueeze(x, 1) # adding a channel dimension

        x = x.permute(0,1,3,2)

        # x = self.bn1(x)
    
        out = self.conv1(x)
        out  = self.max_pool(out)
        out = self.relu(out)

        out = self.conv2(out)
        out  = self.max_pool(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out  = self.max_pool(out)
        out = self.relu(out)
        out = self.point_wise(out)
        # out = out.view(out.shape[0],-1)

        # print("out.shape -> ",out.shape)
        # return
        
        # flattening the output
        out = out.view(out.shape[0],-1)

        out = self.dropout(out)

        out = self.relu(self.fc1(out))
        # out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))
        
        return out