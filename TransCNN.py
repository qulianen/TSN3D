import torch.nn as nn
import torch
from torch.nn.init import *
import torch.nn.functional as F
import numpy as np
import math,copy,time

class Transformer(nn.Module):
    def __init__(self,hidden_dim,N,H,total_size,filters=[1,3,5]):
        super(Transformer, self).__init__()
        self.model = Encoder(EncoderLayer(hidden_dim,MultiHeadedAttention(H,hidden_dim),HAR_CNN(hidden_dim,hidden_dim,filters),0.1),N)

    def forward(self,x,mask=None):
        return self.model(x,mask)

class Trans(nn.Module):
    def __init__(self,hidden_dim,N,H):
        super(Trans, self).__init__()
        self.model = Encoder(EncoderLayer(hidden_dim,MultiHeadedAttention(H,hidden_dim),PositionwiseFeedForward(hidden_dim,hidden_dim*4),0.1),N)

    def forward(self,x,mask = None):
        return self.model(x,mask)

def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module)for _ in range(N)])

class Encoder(nn.Module):
    #N层堆栈
    def __init__(self,layer,N):
        super(Encoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask = None):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self,features,eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):

        mean = x.mean(-1,keepdim=True)

        std = x.std(-1,keepdim = True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2)
        self.size = size

    def forward(self,x,mask = None):
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)


class SublayerConnection(nn.Module):
    # 残差连接
    def __init__(self,size,dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def get_rel_pos(self,x):
        return max(self.k * -1, min(self.k,x))

    def forward(self,query,key,value,mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches,-1, self.h, self.d_k).transpose(1,2)
             for l,x in zip(self.linears,(query,key,value))]

        x, self.attn = attention(query,key,value,mask = mask,dropout = self.dropout)

        x = x.transpose(1,2).contiguous().view(nbatches,-1,self.h * self.d_k)
        return self.linears[-1](x)

def attention(query,key,value,mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores,dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value), p_attn

class HAR_CNN(nn.Module):
    def __init__(self,d_model,d_ff,filters,dropout = 0.1):
        super(HAR_CNN, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(
                                 in_channels=d_model,
                                 out_channels=self.kernel_num,
                                 kernel_size=filter_size,
                                 padding=int((filter_size-1)/2)
                             ))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self,x):
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(x.transpose(-1,-2))
            enc_ = f_map
            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim = 1))
        re = torch.div(torch.sum(torch.cat(enc_outs,1),dim = 1),3)
        encoding = re
        return encoding.transpose(-1,-2)

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)
    
class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        #self.embedding = get_pe(d_model, K).to('cuda')
        #self.register_buffer('pe', self.embedding)
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to('cuda')
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        #print(M)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)