import torch
import torch.nn as nn
import torch.nn.functional as F

batchNorm_momentum = 0.1
num_instruments = 1

class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride) 
        else: 
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 

    def forward(self, x, size=None, isLast=None, skip=None):
        # print(f'x.shape={x.shape}')
        # print(f'target shape = {size}')
        x = self.us(x,output_size=size)
        if not isLast: x = torch.cat((x, skip), 1) 
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))
        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Encoder, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

    def forward(self, x):
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
       
        c1=self.conv1(x3) 
        c2=self.conv2(x2) 
        c3=self.conv3(x1) 
        return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]

class Decoder(nn.Module):
    def __init__(self,ds_ksize, ds_stride):
        super(Decoder, self).__init__()
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride)
        self.d_block4 = d_block(16,num_instruments,True,(3,3),(1,1),ds_ksize, ds_stride)
            

            
    def forward(self, x, s, c=[None,None,None,None]):
        x = self.d_block1(x,s[3],False,c[0])
        x = self.d_block2(x,s[2],False,c[1])
        x = self.d_block3(x,s[1],False,c[2])
        x = self.d_block4(x,s[0],True,c[3])
#         reconsturction = torch.sigmoid(self.d_block4(x,s[0],True,c[3]))
       
        return torch.sigmoid(x)
    
