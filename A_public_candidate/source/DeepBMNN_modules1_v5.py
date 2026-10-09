######################################binarized_modules#################################################
# This code is written by XihongZhou (PhD student)
# date:2020-12-16
# E-mail: g863003a@mails.cc.ehime-u.ac.jp
# Address: Takahashi Lab, Computer System Laboratory, Department of Computer Science, Ehime University
########################################################################################################

import torch
import torch.nn as nn
from binarized_modules import BinarizeLinear

class Out_0_or_1(): # -> 0 or 1
    def __init__(self,flag_0_1 = True):
        self.flag_0_1 = flag_0_1 
    def __call__(self,input):
        if self.flag_0_1 == True:
            input.data = torch.where(input.data <= 0, 0, 1).clone().detach().to(dtype=torch.float)
            return input
        else:
            return input

class MLUTx0(nn.Module):  # 0th column MLUTs 入力層
    def __init__(self,in_size = 784, out_size = 44):
        super(MLUTx0,self).__init__()
        self.in_size = in_size 
        self.out_size = out_size

        self.fc0 = BinarizeLinear(in_size,out_size)
        self.bn0 = nn.BatchNorm1d(out_size)
        self.htanh0 = nn.Hardtanh()
        self.B_0_or_1 = Out_0_or_1()

    def forward(self,input):
        full_connected_out = self.fc0(input)
        full_connected_out = self.bn0(full_connected_out)
        full_connected_out = self.htanh0(full_connected_out)
        full_connected_out = self.B_0_or_1(full_connected_out)

        
        MLUTy0_out = full_connected_out[:,:4]
        MLUTy1_out = full_connected_out[:,4:12]
        MLUTy2_out = full_connected_out[:,12:20]
        MLUTy3_out = full_connected_out[:,20:28]
        MLUTy4_out = full_connected_out[:,28:36]
        MLUTy5_out = full_connected_out[:,36:44]

        return MLUTy0_out,MLUTy1_out,MLUTy2_out,MLUTy3_out,MLUTy4_out,MLUTy5_out

class MLUT_prophase_oddLinear(nn.Module): #1st,3rd column MLUTs
    def __init__(self,in_size =44 ,out_size =44 ):
        super(MLUT_prophase_oddLinear,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        #-----------------Iint deployed neurons MLUT arrays------------------------------
        #********** Iint MLUTy0****************
        self.MLUTy0 = BinarizeLinear(8,8) 
        self.bn_MLUTy0 = nn.BatchNorm1d(8)
        self.htanh_MLUTy0 = nn.Hardtanh()
        self.B_0_or_1_MLUTy0 = Out_0_or_1()
        #********** Iint MLUTy1****************
        self.MLUTy1 = BinarizeLinear(8,8) 
        self.bn_MLUTy1 = nn.BatchNorm1d(8)
        self.htanh_MLUTy1 = nn.Hardtanh()
        self.B_0_or_1_MLUTy1 = Out_0_or_1()
        #********** Iint MLUTy2****************
        self.MLUTy2 = BinarizeLinear(8,8) 
        self.bn_MLUTy2 = nn.BatchNorm1d(8)
        self.htanh_MLUTy2 = nn.Hardtanh()
        self.B_0_or_1_MLUTy2 = Out_0_or_1()
        #********** Iint MLUTy3****************
        self.MLUTy3 = BinarizeLinear(8,8) 
        self.bn_MLUTy3 = nn.BatchNorm1d(8)
        self.htanh_MLUTy3 = nn.Hardtanh()
        self.B_0_or_1_MLUTy3 = Out_0_or_1()
        #********** Iint MLUTy4****************
        self.MLUTy4 = BinarizeLinear(8,8) 
        self.bn_MLUTy4 = nn.BatchNorm1d(8)
        self.htanh_MLUTy4 = nn.Hardtanh()
        self.B_0_or_1_MLUTy4 = Out_0_or_1()
        #********** Iint MLUTy5****************
        self.MLUTy5 = BinarizeLinear(4,4) 
        self.bn_MLUTy5 = nn.BatchNorm1d(4)
        self.htanh_MLUTy5 = nn.Hardtanh()
        self.B_0_or_1_MLUTy5 = Out_0_or_1()

    def forward(self,preMLUTy0,preMLUTy1,preMLUTy2,preMLUTy3,preMLUTy4,preMLUTy5):

        MLUTy0_input = torch.cat((preMLUTy0[:,-4:],preMLUTy1[:,:4]),dim=1) #8
        MLUTy1_input = torch.cat((preMLUTy1[:,-4:],preMLUTy2[:,:4]),dim=1) #8
        MLUTy2_input = torch.cat((preMLUTy2[:,-4:],preMLUTy3[:,:4]),dim=1) #8
        MLUTy3_input = torch.cat((preMLUTy3[:,-4:],preMLUTy4[:,:4]),dim=1) #8
        MLUTy4_input = torch.cat((preMLUTy4[:,-4:],preMLUTy5[:,:4]),dim=1) #8
        MLUTy5_input = preMLUTy5[:,-4:]                                    #4

        #-----------------Forward deployed neurons MLUT arrays------------------------------
        #********** Forward MLUTy0**************
        MLUTy0_out = self.MLUTy0(MLUTy0_input)
        MLUTy0_out = self.bn_MLUTy0(MLUTy0_out)
        MLUTy0_out = self.htanh_MLUTy0(MLUTy0_out)
        MLUTy0_out = self.B_0_or_1_MLUTy0(MLUTy0_out)
        #********** Forward MLUT1**************
        MLUTy1_out = self.MLUTy1(MLUTy1_input)
        MLUTy1_out = self.bn_MLUTy1(MLUTy1_out)
        MLUTy1_out = self.htanh_MLUTy1(MLUTy1_out)
        MLUTy1_out = self.B_0_or_1_MLUTy1(MLUTy1_out)
        #********** Forward MLUTy2**************
        MLUTy2_out = self.MLUTy2(MLUTy2_input)
        MLUTy2_out = self.bn_MLUTy2(MLUTy2_out)
        MLUTy2_out = self.htanh_MLUTy2(MLUTy2_out)
        MLUTy2_out = self.B_0_or_1_MLUTy2(MLUTy2_out)
        #********** Forward MLUTy3**************
        MLUTy3_out = self.MLUTy3(MLUTy3_input)
        MLUTy3_out = self.bn_MLUTy3(MLUTy3_out)
        MLUTy3_out = self.htanh_MLUTy3(MLUTy3_out)
        MLUTy3_out = self.B_0_or_1_MLUTy3(MLUTy3_out)
        #********** Forward MLUTy4**************
        MLUTy4_out = self.MLUTy4(MLUTy4_input)
        MLUTy4_out = self.bn_MLUTy4(MLUTy4_out)
        MLUTy4_out = self.htanh_MLUTy4(MLUTy4_out)
        MLUTy4_out = self.B_0_or_1_MLUTy4(MLUTy4_out)
        #********** Forward MLUTy5**************
        MLUTy5_out = self.MLUTy5(MLUTy5_input)
        MLUTy5_out = self.bn_MLUTy5(MLUTy5_out)
        MLUTy5_out = self.htanh_MLUTy5(MLUTy5_out)
        MLUTy5_out = self.B_0_or_1_MLUTy5(MLUTy5_out)

        return MLUTy0_out,MLUTy1_out,MLUTy2_out,MLUTy3_out,MLUTy4_out,MLUTy5_out

class MLUT_prophase_evenLinear(nn.Module): #2nd,4th column MLUTs
    def __init__(self,in_size = 44,out_size = 44):
        super(MLUT_prophase_evenLinear,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        #-----------------Iint deployed neurons MLUT arrays------------------------------
        #********** Iint MLUTy0****************
        self.MLUTy0 = BinarizeLinear(4,4) 
        self.bn_MLUTy0 = nn.BatchNorm1d(4)
        self.htanh_MLUTy0 = nn.Hardtanh()
        self.B_0_or_1_MLUTy0 = Out_0_or_1()
        #********** Iint MLUTy1****************
        self.MLUTy1 = BinarizeLinear(8,8) 
        self.bn_MLUTy1 = nn.BatchNorm1d(8)
        self.htanh_MLUTy1 = nn.Hardtanh()
        self.B_0_or_1_MLUTy1 = Out_0_or_1()
        #********** Iint MLUTy2****************
        self.MLUTy2 = BinarizeLinear(8,8) 
        self.bn_MLUTy2 = nn.BatchNorm1d(8)
        self.htanh_MLUTy2 = nn.Hardtanh()
        self.B_0_or_1_MLUTy2 = Out_0_or_1()
        #********** Iint MLUTy3****************
        self.MLUTy3 = BinarizeLinear(8,8) 
        self.bn_MLUTy3 = nn.BatchNorm1d(8)
        self.htanh_MLUTy3 = nn.Hardtanh()
        self.B_0_or_1_MLUTy3 = Out_0_or_1()
        #********** Iint MLUTy4****************
        self.MLUTy4 = BinarizeLinear(8,8) 
        self.bn_MLUTy4 = nn.BatchNorm1d(8)
        self.htanh_MLUTy4 = nn.Hardtanh()
        self.B_0_or_1_MLUTy4 = Out_0_or_1()
        #********** Iint MLUTy5****************
        self.MLUTy5 = BinarizeLinear(8,8) 
        self.bn_MLUTy5 = nn.BatchNorm1d(8)
        self.htanh_MLUTy5 = nn.Hardtanh()
        self.B_0_or_1_MLUTy5 = Out_0_or_1()

    def forward(self,preMLUTy0,preMLUTy1,preMLUTy2,preMLUTy3,preMLUTy4,preMLUTy5):

        MLUTy0_input = preMLUTy0[:,:4]                                    #4
        MLUTy1_input = torch.cat((preMLUTy0[:,-4:],preMLUTy1[:,:4]),dim=1) #8
        MLUTy2_input = torch.cat((preMLUTy1[:,-4:],preMLUTy2[:,:4]),dim=1) #8
        MLUTy3_input = torch.cat((preMLUTy2[:,-4:],preMLUTy3[:,:4]),dim=1) #8
        MLUTy4_input = torch.cat((preMLUTy3[:,-4:],preMLUTy4[:,:4]),dim=1) #8
        MLUTy5_input = torch.cat((preMLUTy4[:,-4:],preMLUTy5[:,:4]),dim=1) #8
        #-----------------Forward deployed neurons MLUT arrays------------------------------
        #********** Forward MLUTy0**************
        MLUTy0_out = self.MLUTy0(MLUTy0_input)
        MLUTy0_out = self.bn_MLUTy0(MLUTy0_out)
        MLUTy0_out = self.htanh_MLUTy0(MLUTy0_out)
        MLUTy0_out = self.B_0_or_1_MLUTy0(MLUTy0_out)
        #********** Forward MLUT1**************
        MLUTy1_out = self.MLUTy1(MLUTy1_input)
        MLUTy1_out = self.bn_MLUTy1(MLUTy1_out)
        MLUTy1_out = self.htanh_MLUTy1(MLUTy1_out)
        MLUTy1_out = self.B_0_or_1_MLUTy1(MLUTy1_out)
        #********** Forward MLUTy2**************
        MLUTy2_out = self.MLUTy2(MLUTy2_input)
        MLUTy2_out = self.bn_MLUTy2(MLUTy2_out)
        MLUTy2_out = self.htanh_MLUTy2(MLUTy2_out)
        MLUTy2_out = self.B_0_or_1_MLUTy2(MLUTy2_out)
        #********** Forward MLUTy3**************
        MLUTy3_out = self.MLUTy3(MLUTy3_input)
        MLUTy3_out = self.bn_MLUTy3(MLUTy3_out)
        MLUTy3_out = self.htanh_MLUTy3(MLUTy3_out)
        MLUTy3_out = self.B_0_or_1_MLUTy3(MLUTy3_out)
        #********** Forward MLUTy4**************
        MLUTy4_out = self.MLUTy4(MLUTy4_input)
        MLUTy4_out = self.bn_MLUTy4(MLUTy4_out)
        MLUTy4_out = self.htanh_MLUTy4(MLUTy4_out)
        MLUTy4_out = self.B_0_or_1_MLUTy4(MLUTy4_out)
        #********** Forward MLUTy5**************
        MLUTy5_out = self.MLUTy5(MLUTy5_input)
        MLUTy5_out = self.bn_MLUTy5(MLUTy5_out)
        MLUTy5_out = self.htanh_MLUTy5(MLUTy5_out)
        MLUTy5_out = self.B_0_or_1_MLUTy5(MLUTy5_out)

        
        #こっちのMLUTyn_outはMLUTynからの出力
        return MLUTy0_out,MLUTy1_out,MLUTy2_out,MLUTy3_out,MLUTy4_out,MLUTy5_out
