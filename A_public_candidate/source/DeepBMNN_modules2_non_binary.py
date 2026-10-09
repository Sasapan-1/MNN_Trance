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

class MLUTx0_48(nn.Module):  # 0th column MLUTs 入力層
    def __init__(self,in_size = 784, out_size = 48):
        super(MLUTx0_48,self).__init__()
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
        
        
        MLUTy0_out = full_connected_out[:,:8]
        MLUTy1_out = full_connected_out[:,8:16]
        MLUTy2_out = full_connected_out[:,16:24]
        MLUTy3_out = full_connected_out[:,24:32]
        MLUTy4_out = full_connected_out[:,32:40]
        MLUTy5_out = full_connected_out[:,40:48]

        return MLUTy0_out,MLUTy1_out,MLUTy2_out,MLUTy3_out,MLUTy4_out,MLUTy5_out

class MLUT_input_Linear1_b(nn.Module):
    def __init__(self,in_size = 48,out_size = 40):
        super(MLUT_input_Linear1_b,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        #-----------------Iint deployed neurons MLUT arrays------------------------------
        #********** Iint MLUTy0****************
        self.MLUTy0 = BinarizeLinear(8,4) 
        self.bn_MLUTy0 = nn.BatchNorm1d(4)
        self.htanh_MLUTy0 = nn.Hardtanh()
        self.o1_MLUTy0 = Out_0_or_1()
        #********** Iint MLUTy1****************
        self.MLUTy1 = BinarizeLinear(8,8) 
        self.bn_MLUTy1 = nn.BatchNorm1d(8)
        self.htanh_MLUTy1 = nn.Hardtanh()
        self.o1_MLUTy1 = Out_0_or_1()
        #********** Iint MLUTy2****************
        self.MLUTy2 = BinarizeLinear(8,8) 
        self.bn_MLUTy2 = nn.BatchNorm1d(8)
        self.htanh_MLUTy2 = nn.Hardtanh()
        self.o1_MLUTy2 = Out_0_or_1()
        #********** Iint MLUTy3****************
        self.MLUTy3 = BinarizeLinear(8,8) 
        self.bn_MLUTy3 = nn.BatchNorm1d(8)
        self.htanh_MLUTy3 = nn.Hardtanh()
        self.o1_MLUTy3 = Out_0_or_1()
        #********** Iint MLUTy4****************
        self.MLUTy4 = BinarizeLinear(8,8) 
        self.bn_MLUTy4 = nn.BatchNorm1d(8)
        self.htanh_MLUTy4 = nn.Hardtanh()
        self.o1_MLUTy4 = Out_0_or_1()
        #********** Iint MLUTy4****************
        self.MLUTy5 = BinarizeLinear(8,4) 
        self.bn_MLUTy5 = nn.BatchNorm1d(4)
        self.htanh_MLUTy5 = nn.Hardtanh()
        self.o1_MLUTy5 = Out_0_or_1()


    def forward(self,MLUTy0_in,MLUTy1_in,MLUTy2_in,MLUTy3_in,MLUTy4_in,MLUTy5_in):

       #入力層なので前のLUTは存在しない
        MLUTy0_input = MLUTy0_in #8
        MLUTy1_input = MLUTy1_in #8
        MLUTy2_input = MLUTy2_in #8
        MLUTy3_input = MLUTy3_in #8
        MLUTy4_input = MLUTy4_in #8
        MLUTy5_input = MLUTy5_in #8
        #-----------------Forward deployed neurons MLUT arrays------------------------------
        #********** Forward MLUTy0**************
        MLUTy0_out = self.MLUTy0(MLUTy0_input)
        MLUTy0_out = self.bn_MLUTy0(MLUTy0_out)
        MLUTy0_out = self.htanh_MLUTy0(MLUTy0_out)
        MLUTy0_out = self.o1_MLUTy0(MLUTy0_out)
        #********** Forward MLUTy1**************
        MLUTy1_out = self.MLUTy1(MLUTy1_input)
        MLUTy1_out = self.bn_MLUTy1(MLUTy1_out)
        MLUTy1_out = self.htanh_MLUTy1(MLUTy1_out)
        MLUTy1_out = self.o1_MLUTy1(MLUTy1_out)
        #********** Forward MLUTy2**************
        MLUTy2_out = self.MLUTy2(MLUTy2_input)
        MLUTy2_out = self.bn_MLUTy2(MLUTy2_out)
        MLUTy2_out = self.htanh_MLUTy2(MLUTy2_out)
        MLUTy2_out = self.o1_MLUTy2(MLUTy2_out)
        #********** Forward MLUTy3**************
        MLUTy3_out = self.MLUTy3(MLUTy3_input)
        MLUTy3_out = self.bn_MLUTy3(MLUTy3_out)
        MLUTy3_out = self.htanh_MLUTy3(MLUTy3_out)
        MLUTy3_out = self.o1_MLUTy3(MLUTy3_out)
        #********** Forward MLUTy4**************
        MLUTy4_out = self.MLUTy4(MLUTy4_input)
        MLUTy4_out = self.bn_MLUTy4(MLUTy4_out)
        MLUTy4_out = self.htanh_MLUTy4(MLUTy4_out)
        MLUTy4_out = self.o1_MLUTy4(MLUTy4_out)
        #********** Forward MLUTy5**************
        MLUTy5_out = self.MLUTy5(MLUTy5_input)
        MLUTy5_out = self.bn_MLUTy5(MLUTy5_out)
        MLUTy5_out = self.htanh_MLUTy5(MLUTy5_out)
        MLUTy5_out = self.o1_MLUTy5(MLUTy5_out)

        return MLUTy0_out,MLUTy1_out,MLUTy2_out,MLUTy3_out,MLUTy4_out,MLUTy5_out
