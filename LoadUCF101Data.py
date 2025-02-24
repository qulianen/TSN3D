""""
加载处理好的UCF101数据
"""
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image


# 超参数
TRAIN_BATCH_SIZE = 6  # 训练集的批量
TEST_BATCH_SIZE = 6  # 测试集的批量
SAMPLE_FRAME_NUM = 16  # 选择10帧光流来堆叠，送入光流卷积网络中
FLOW_INTERVAL = 2  # 光流图像数据的生成间隔，默认为2，与generate_rgb_and_flow.py代码flow_save_interval中的一致

# 将classInd.txt中的动作类别添加到列表中
classInd = []
with open('./ucfTrainTestlist/classInd.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        idx = line[:-1].split(' ')[0]
        className = line[:-1].split(' ')[1]
        classInd.append(className)

# 将trainlist01.txt中划分的训练集动作视频存入TrainVideoNameList列表中
TrainVideoNameList = []
with open('./ucfTrainTestlist/trainlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split('.')[0]
        video_name = video_name.split('/')[1]
        TrainVideoNameList.append(video_name)


# 将testlist01.txt中划分的测试集动作视频存入TestVideoNameList列表中
TestVideoNameList = []
with open('./ucfTrainTestlist/testlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split('.')[0]
        video_name = video_name.split('/')[1]
        TestVideoNameList.append(video_name)


# 定义一个UCF101Data数据加载的类
class UCF101Data(Dataset):

    def __init__(self, RBG_root, OpticalFlow_root, data_class="train", transform=None, action_num=3, flow_interval=2):
        """
        :param RBG_root:  RGB数据集的文件路径
        :param OpticalFlow_root:  OpticalFlow_root数据集的文件路径
        :param data_class: 需要加载的数据类型
        :param transform: 是否对数据根据transform的操作
        :param action_num: 动作的类别个数
        """
        self.filenames = []
        self.transform = transform
        self.num = action_num

        for i in range(0, self.num):    # dmk修改，本类别数为2
            # 依次读取classInd中的动作名字，并依次生成光流数集中的每类动作的目录
            OpticalFlow_class_path = OpticalFlow_root + '/' + classInd[i]

            # 依次读取classInd中的动作名字，并依次生成RGB数集中的每类动作的目录
            RGB_class_path = RBG_root + '/' + classInd[i]

            # 将OpticalFlow_class_path路径下的文件生成列表，
            # 并与TrainVideoNameList/TestVideoNameList列表中内容求交集，即获取第i类动作需要训练/测试的全部动作视频
            if data_class == "train":
                TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_class_path)).intersection(set(TrainVideoNameList)))
            if data_class == "test":
                TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_class_path)).intersection(set(TestVideoNameList)))

            # 依次遍历第i个OpticalFlow_class_path路径下的所有视频文件
            for video_dir in os.listdir(OpticalFlow_class_path):
                # 判断此时路径下的视频文件是否在TrainOrTest_VideoNameList列表中
                # 如果在的话，就生成single_OpticalFlow_video_path和signel_RGB_video_path的路径
                if video_dir in TrainOrTest_VideoNameList:
                    single_OpticalFlow_video_path = OpticalFlow_class_path + '/' + video_dir
                    signel_RGB_video_path = RGB_class_path + '/' + video_dir

                    # 加载single_OpticalFlow_video_path中的所有光流文件名字，并放入fram_list列表中
                    frame_list = os.listdir(single_OpticalFlow_video_path)

                    # 根据当前视频生成的每个光流图像的索引值来对frame_list列表数据进行排序，也即当前视频每帧生成的光流图像顺序
                    # 如v_ApplyEyeMakeup_g08_c01_2_x.jpg中的……_2_……的值就是排序位
                    frame_list.sort(key=lambda x: int(x.split("_")[-2]))
                    # 在fram_list列表中随机生成一个光流图像的索引值，但此索引值一定是从光流图像的x方向光流图像开始的
                    # 首先在【当前视频的光流总个数-SAMPLE_FRAME_NUM * 2 + 1】的范围内随机生成一个索引值，
                    # 这样做的目的是为了防止最后堆叠的光流索引超出当前视频的总光流图像的范围，保证每次不管随机生成几，都能产生SAMPLE_FRAME_NUM个堆叠光流
                    ran_frame_idx = np.random.randint(0, len(frame_list) - (SAMPLE_FRAME_NUM * 2) + 1)
                    # 如果不是从x值开始的，则继续生成新索引值，直到符合规定为止
                    while ran_frame_idx % flow_interval != 0:
                        ran_frame_idx = np.random.randint(0, len(frame_list) - SAMPLE_FRAME_NUM * 2 + 1)
                    # 在这里堆叠SAMPLE_FRAME_NUM个光流图像的路径在接下来的列表中，包括x和y方向上的光流，所以共20个
                    stacked_OpticalFlow_image_path = []
                    for j in range(ran_frame_idx, ran_frame_idx + SAMPLE_FRAME_NUM * 2):
                        OpticalFlow_image_path = single_OpticalFlow_video_path + '/' + frame_list[j]
                        stacked_OpticalFlow_image_path.append(OpticalFlow_image_path)
                    #print(stacked_OpticalFlow_image_path)


                    # 随机从当前视频中的提取一帧RGB图像
                    rgb_frame_list = os.listdir(signel_RGB_video_path)
                    rgb_frame_list.sort(key=lambda x: int((x.split("_")[-1]).split(".")[0]))
                    randm_rgb = np.random.randint(0, len(rgb_frame_list)- (SAMPLE_FRAME_NUM//2) + 1)
                    #RGB_image_path = signel_RGB_video_path + '/' + rgb_frame_list[randm_rgb]
                    stacked_RGB_image_path = []
                    for j in range(randm_rgb,randm_rgb + SAMPLE_FRAME_NUM//2):
                        RGB_image_path = signel_RGB_video_path + '/' + rgb_frame_list[randm_rgb]
                        stacked_RGB_image_path.append(RGB_image_path)

                    # 将上面生成的数据，弄成这样(RGB_image_path, stacked_OpticalFlow_image_path, label)的形式，并添加到filenames列表中
                    # 这里即1个视频数据的RGB+Flow+标签生成
                    self.filenames.append((stacked_RGB_image_path, stacked_OpticalFlow_image_path, i))

        self.len = len(self.filenames)  # 得到最后filenames列表的长度


    # 重写Dateset类中的__getitem__方法
    # 根据index的值，从filenames列表中，获取此索引对应的列表内容(RGB+FLOW+label)
    def __getitem__(self, index):
        stacked_RGB_image_path, stacked_OpticalFlow_image_path, label = self.filenames[index]
        # 创建一个SAMPLE_FRAME_NUM * 2, 224, 224的张量
        #stacked_OpticalFlow_image = torch.empty(SAMPLE_FRAME_NUM * 2, 224, 224)
        stacked_RGB_image = torch.empty(SAMPLE_FRAME_NUM//2,3,112,112)
        stacked_OpticalFlow_image = torch.empty(2, SAMPLE_FRAME_NUM , 112, 112)
        idx = 0
        idy = 0
        idr = 0
        #print(stacked_OpticalFlow_image.size())
        
        for i in stacked_RGB_image_path:
            RGB_image = Image.open(i)
            if self.transform is not None:
                RGB_image = self.transform(RGB_image)
            #print(RGB_image)
            stacked_RGB_image[idr,: :,:] = RGB_image[0,:,:]
            #print(stacked_RGB_image)
            idr +=1

        # 依次处理stacked_OpticalFlow_image_path列表中的所有光流图像(20张)
        for i in stacked_OpticalFlow_image_path:
            # 用Image.open()打开第i个路径的光流图像
            OpticalFlow_image = Image.open(i)
            
            ifxy = i.split('/')[5]
            ifxy = ifxy.split('_')[5]
            ifxy = ifxy.split('.')[0]

            # 是否使用transform对图像大小和像素值进行修改，判断
            if self.transform is not None:
                OpticalFlow_image = self.transform(OpticalFlow_image)
            if(ifxy == 'x'):
                stacked_OpticalFlow_image[0,idx, :, :] = OpticalFlow_image[0, :, :]
            #stacked_OpticalFlow_image[:, idx, :, :] = OpticalFlow_image[:, :, :]
                idx += 1
            if(ifxy == 'y'):
                stacked_OpticalFlow_image[1,idy, :, :] = OpticalFlow_image[0, :, :]
                idy += 1
                

        # 用Image.open()打开RGB_image_path第路径的RGB图像
        #RGB_image = Image.open(RGB_image_path)
        # 是否使用transform对图像大小和像素值进行修改
        #if self.transform is not None:

        #    RGB_image = self.transform(RGB_image)

        return stacked_RGB_image, stacked_OpticalFlow_image, label

    # 得到数据集的总长度
    def __len__(self):
        return self.len


# transforms.Compose()将以下操作放在一起
# 依次是Resize成256*256；随机裁剪224*224；
# 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
transform = transforms.Compose([transforms.Resize([128,171]), transforms.RandomCrop([112,112]), transforms.ToTensor()])
"""
1.加载UCF101数据集中划分的训练数据集
2.RBG_root:RGB数据集路径；OpticalFlow_root:光流数据集路径；data_class:加载train/test集数据；
transform:对图像数据进行一些列变化操作；action_num:动作的类别
3.下面的test数据类似
"""
trainset = UCF101Data(RBG_root='./data/RGB', OpticalFlow_root='./data/OpticalFlow', data_class="train", transform=transform, action_num=6)
# 利用DataLoader()函数,将数据以批量的大小加载，存放
trainset_loader = DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

# 加载UCF101数据集中划分的测试集数据集
testset = UCF101Data(RBG_root='./data/RGB', OpticalFlow_root='./data/OpticalFlow', data_class="test", transform=transform, action_num=6)
testset_loader = DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)


if __name__ == '__main__':

    print(len(trainset_loader.dataset))
    for i, data in enumerate(trainset_loader):
        # 加载bach_size个数据
        RGB_images, OpticalFlow_images, label = data
        if i == 0:
            print(RGB_images.size())
            print(OpticalFlow_images.size())
            print(label.size())
        if i == (len(trainset_loader)-1):
            print(RGB_images.size())
            print(OpticalFlow_images.size())
            print(label.size())
2
