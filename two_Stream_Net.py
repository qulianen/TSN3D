"""
双流网络模型，resnet101主干网络
"""
import torch
import torch.nn as nn
import torchvision.models as models
import LoadUCF101Data
#from TransCNN import Transformer,Trans,Gaussian_Position
from Trans import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
0
#torch.Tensor.purmute
# 定义光流网络类
class OpticalFlowStreamNet(nn.Module):
    def __init__(self):
        super(OpticalFlowStreamNet, self).__init__()

        self.OpticalFlow_stream = models.resnet101()  # 模型选用resnet101
        # 改变resnet101的第一层和最后一层
        self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=6, padding=3,bias=False)
        self.OpticalFlow_stream.fc = nn.Linear(in_features=2048, out_features=6)

    def forward(self, x):
        streamOpticalFlow_out = self.OpticalFlow_stream(x)
        #print(streamOpticalFlow_out.size())
        return streamOpticalFlow_out

class c3dOpitical(nn.Module):
    def __init__(self,num_class,pretrained = False):
        super(c3dOpitical, self).__init__()
        self.conv1 = nn.Conv3d(2,64,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))

        self.conv2 = nn.Conv3d(64,128,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))

        self.conv3a = nn.Conv3d(128,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv3b = nn.Conv3d(256,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))

        self.conv4a = nn.Conv3d(256,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv4b = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))

        self.conv5a = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv5b = nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=(0,1,1))

        self.fc6 = nn.Linear(8192,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,num_class)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self,x):
        #print(x.size())
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        #print(x.size())

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        #print(x.size())

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        #print(x.size())

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        #print(x.size())

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        #print(x.size())

        x = x.view(-1,8192)
        #print(x.size())
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        #print(x.size())
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        #print(x.size())

        logits = self.fc8(x)
        #print(logits.size())

        return logits


    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load('./model')
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



# # 定RGB网络类
class RGBStreamNet(nn.Module):
    def __init__(self):
        super(RGBStreamNet, self).__init__()
        # 模型选用resnet101,并使用预训练模型
        self.RGB_stream = models.resnet101(pretrained=True)
        self.RGB_stream.fc = nn.Linear(in_features=2048, out_features=6)


    def forward(self, x):
        streamRGB_out = self.RGB_stream(x)
        #print(streamRGB_out.size())
        return streamRGB_out

#Transformer
class TransRGBNet(nn.Module):
    def __init__(self):
        super(TransRGBNet, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim = 1)
        
        self.PatchEmbedding = nn.Sequential(
            Rearrange('b c (h p1)(w p2) -> b (h w)(p1 p2 c)',p1 = 16,p2 = 16),
            nn.Linear(768,768)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, 768))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.dropout = nn.Dropout(0.)
        
        
        self.transformer = Transformer(768,6,64,500)
        self.kernel_num = 128
        #self.kernel_num_v = 16

        self.filter_sizes = [10,40]
        #self.filter_sizes_v = [2,4]
        #self.pos_encoding = Gaussian_Position(112, 500, 10)

        #self.v_transformer = Trans(2016,1,126)
        self.v_transformer = None
        self.dense = torch.nn.Linear(112,6)
        #self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), 6)
        
        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes),6)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        #self.encoder_v = []

        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=768,
                                             out_channels=self.kernel_num,
                                             kernel_size=filter_size).to('cuda'))
            self.encoders.append(self.__getattr__(enc_attr_name))
            
        #for i, filter_size in enumerate(self.filter_sizes_v):
        #    enc_attr_name_v = "encoder_v_%d" % i
        #    self.__setattr__(enc_attr_name_v,
        #                     torch.nn.Conv1d(in_channels=2016,
         #                                    out_channels=self.kernel_num_v,
        #                                     kernel_size=filter_size).to('cuda'))
        #    self.encoder_v.append(self.__getattr__(enc_attr_name_v))


    def _aggregate(self,o,v = None):
        enc_outs = []
        enc_outs_v = []

        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1,-2))
            enc_ = nn.functional.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = nn.functional.max_pool1d(enc_,kernel_size = k_h)
            enc_ = enc_.squeeze(dim = -1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs,1))

        q_re = nn.functional.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = nn.functional.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = nn.functional.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = nn.functional.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self,data):
        x = self.PatchEmbedding(data)
        b,n,_ = x.shape
        cls_tokens = repeat(self.cls_token,'() n d -> b n d',b = b)
        x = torch.cat((cls_tokens,x),dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        #d1 = data.size(dim = 0)
        #d3 = data.size(2)
        #print(data.size())
        #x = data.unsqueeze(-2)
        #x = data.view(d1, -1 ,4,d3)
        #print(x.size())
        #x = torch.sum(x,dim = -2).squeeze(-2)
        #x = torch.div(x, 4)
       # print(x.size())
        #x = self.pos_encoding(x)
        x = self.transformer(x)

        re = self._aggregate(x)
        re = self.dense2(re)
        #if self.v_transformer is not None:
            #y = data.view(-1, 2016, 4,28 )       
            #y = torch.sum(y, dim=-2).squeeze(-2)
            #y = y.transpose(-1, -2)
            
        #    y = self.v_transformer(y)
        #    re = self._aggregate(x, y)
        #    re = self.dense(re)
            #predict = self.softmax(self.dense(re))
       # else:
        #    re = self._aggregate(x)
        #    re = self.dense2(re)
            #predict = self.softmax(self.dense2(re))
        #predict = self.softmax(self.dense2(re))
        #re = self.dense2(re)
        
        return re
        

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 8, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )


        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

# 定义双流网络类
class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()

        #self.rgb_branch = RGBStreamNet()
        #self.rgb_branch = TransRGBNet()
        self.rgb_branch = ViViT(112,8,6,8)


        #self.opticalFlow_branch = OpticalFlowStreamNet()
        self.opticalFlow_branch = c3dOpitical(num_class=6)
        

    def forward(self, x_rgb, x_opticalFlow):
        rgb_out = self.rgb_branch(x_rgb)
        
        opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
        #print(opticalFlow_out)
        #print(rgb_out)
        

        #out = torch.concat([rgb_out,opticalFlow_out])
        #final_out = nn.Softmax(dim=1)(out)

        return final_out
