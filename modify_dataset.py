import os
os.environ['HF_HOME'] = '/mnt/nfs/file_server/public/xy/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch 
from torchvision import transforms
from transformers import  CLIPTokenizer
from PIL import Image
import cv2
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# Dataset
class   DAVIS_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, tokenizer_2, size=1024, center_crop=True, max_expand = True,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, 
                 image_root_path="/home/lijiahui/dataset/DAVIS_2016",
                 file_path= "/home/lijiahui/dataset/DAVIS_2016/ImageSets/1080p/train.txt", ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop   #？
        self.max_expand = max_expand
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.transform = transforms.Compose([
            transforms.Resize((self.size,self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                #       0~1
            transforms.Normalize([0.5], [0.5]),]) #归一化 -1~1

        # list of dict: [{"image_path": "/JPEGImages/1080p/bear/00000.jpg", "mask_path": "/Annotations/1080p/bear/00000.png", "text": "A dog"}]
        data = []
        with open(file_path, "r") as lines:
            for idx, line in enumerate(lines):          # 逐行读取文件内容
                item = {}
                img_path, mask_path = line.strip().split() # 去除行末的换行符，并按空格进行分割
                word_a = img_path.split("/")[3]
                word_b = mask_path.split("/")[3]
                if word_a != word_b:
                    print(":EOORO:",idx, line)
                    raise ValueError("Dataset img&mask not match！")
                else:
                    text = 'a photo of a ' + word_a
                    item['text'] = text
                    item['image_path'] = img_path
                    item['mask_path'] = mask_path
                    data.append(item)
        self.data = data
        print("dataset intialize success")
    
    # 提取高频边缘信息图作为controlnet——cond：
    def sobel(self, img_array, mask_array, thresh = 50, erode_kernel_size = 3,iterations = 2, Ksize = 3):
        #Calculating the high-frequency map.
        H,W = img_array.shape[0], img_array.shape[1]
        img = cv2.resize(img_array,(256,256))
        mask = (cv2.resize(mask_array,(256,256)) > 0.5).astype(np.uint8)
        kernel = np.ones((erode_kernel_size,erode_kernel_size),np.uint8)
        mask = cv2.erode(mask, kernel, iterations = iterations) #2次形态学腐蚀：mask向内收缩
        
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize) 
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize) 
        sobel_X = cv2.convertScaleAbs(sobelx) #计算图像在水平方向的梯度
        sobel_Y = cv2.convertScaleAbs(sobely) #计算图像在垂直方向的梯度
        #两个梯度图的绝对值相加，得到图像的梯度强度图
        scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
        # 梯度强度图与掩码相乘，以消除无关区域的干扰
        scharr = np.max(scharr,-1) * mask 
        
        scharr[scharr < thresh] = 0.0
        scharr = np.stack([scharr,scharr,scharr],-1)
        scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
        scharr = cv2.resize(scharr,(W,H))
        return scharr

    # 1. 获得mask的方框坐标bbox = (y1,y2,x1,x2)
    def get_bbox_from_mask(self, mask):
        h,w = mask.shape[0],mask.shape[1]
        if mask.sum() < 10:
            return 0,h,0,w # 无效mask：返回整个图像的坐标
        rows = np.any(mask,axis=1) # 在水平方向：检测mask的有效区域所在列
        cols = np.any(mask,axis=0) # 检测mask的有效区域所在行
        y1,y2 = np.where(rows)[0][[0,-1]] #取出有效区域所在列的第一个和最后一个索引：左右边界坐标
        x1,x2 = np.where(cols)[0][[0,-1]] #取出有效区域所在行的第一个和最后一个索引：左右边界坐标
        return (y1,y2,x1,x2)

    # 2. change box to squre, if can't change: we first the longest side then crop
    def crop_box2squre(self, image, box):
        crop_flag = 0
        H,W = image.shape[0], image.shape[1]
        y1,y2,x1,x2 = box
        box_height ,box_width = y2-y1, x2-x1
        print(box,box_height,box_width)

        if box_width == box_height:
            print("本身mask就是squre")
            return (y1,y2,x1,x2),crop_flag
        
        side_length = max(box_width, box_height)
        if side_length > min(H, W):
            print("mask 形状不适合 to square,需要中心裁剪")
            side_length = min(H,W)
            crop_side = 'x' if box_width > box_height else 'y'  # 哪边超长就裁剪哪边
            if crop_side =="x":
                x1= x1+(box_width - side_length)//2
                x2= x2-(box_width - side_length)//2
            else:
                y1 = y1+(box_height - side_length)//2
                y2 = y2-(box_height - side_length)//2
            crop_flag = 1
        else:
            change_side = "y" if box_width > box_height else "x" # 哪边短就变哪边
            if change_side =="x" :
                if x1 >= (side_length - box_width)//2 and (W-x2) >= ((side_length - box_width)//2+1):
                    x1= x1 - (side_length - box_width)//2
                    x2= x1 + side_length
                elif x1 < (side_length - box_width)//2 and (x2 + (side_length - box_width- x1)) <= W:
                    x2= x2 + (side_length - box_width - x1)
                    x1= 0
                elif (x1 - (side_length - box_width- (W - x2)))>=0 and (W- x2) < (side_length - box_width)//2:
                    x1 = x1 - (side_length -box_width- (W - x2))
                    x2 = W
            else:
                if y1 >= (side_length - box_height)//2 and (H-y2) >= ((side_length - box_height)//2 +1):
                    # print("1")
                    y1= y1 - (side_length - box_height)//2
                    y2= y1+ side_length
                elif y1 < (side_length - box_height)//2 and (y2 + (side_length - box_height- y1)) <= H:
                    # print("2")
                    y2= y2 + (side_length - box_height - y1)
                    y1= 0
                elif (y1 - (side_length - box_height- (H - y2)))>=0 and (H- y2) < (side_length - box_height)//2:
                    # print("3")
                    y1 = y1 - (side_length - box_height - (H - y2))
                    y2 = H
                else:
                    print('error!!!!')
                    return None,crop_flag
        if (y2-y1) == (x2-x1):
            if y2 > H or x2 > W:
                print("error!!!")
                print("side_length:",side_length,"change_side:",change_side)
                print("new h:",y2-y1,"new w:",x2-x1)
                print("yyxx:",y1,y2,x1,x2)
                print("crop_flag:",crop_flag)
                return None,crop_flag
            else:
                return (y1,y2,x1,x2),crop_flag
        else:
            print("something error, output is not a square!!!!!!!!")
            print("side_length:",side_length,"change_side:",change_side)
            print("new h:",y2-y1,"new w:",x2-x1)
            print("yyxx:",y1,y2,x1,x2)
            print("crop_flag:",crop_flag)
            return None,crop_flag
            
    # 3. 自适应：expand squre box 
    def random_expand_squre_bbox(self, ori_mask,yyxx, max_expand=True):
        H,W = ori_mask.shape[0], ori_mask.shape[1]   #H: 1080 W: 1920
        y1,y2,x1,x2 = yyxx

        if (y2-y1) != (x2-x1):
            return "ERROR!!!!!!!!before this, u should make the bbox to square!!!!"
        # 在对角线两个方向进行expand：
        if max_expand ==False:
            expand_a = 0 if min(x1,y1) == 0 else np.random.randint( 0 , min(x1, y1) )
            expand_b = 0 if min((W-x2),(H-y2)) ==0 else np.random.randint( 0 , min((W-x2),(H-y2)))
        else:
            expand_a = min(x1, y1) 
            expand_b = min((W-x2),(H-y2))

        x1 = x1 - expand_a
        y1 = y1 - expand_a
        x2 = x2 + expand_b
        y2 = y2 + expand_b
        if (y2-y1) != (x2-x1):
            return "ERROR!!!!!!!!check and make the bbox to square!!!!"
        return (y1,y2,x1,x2)

    # 1-2-3:
    def crop_img_mask_squre(self, img, mask, max_expand):
        box = self.get_bbox_from_mask(mask) 
        box, crop_flag = self.crop_box2squre(img, box) 
        box = self.random_expand_squre_bbox(mask,box, max_expand) 
        return box,crop_flag


    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_path"]
        mask_file = item["mask_path"]
        text = item["text"]
        
        # read image
        raw_file = self.image_root_path + image_file

        raw_image = Image.open(raw_file)
        mask = Image.open(self.image_root_path + mask_file)
        
        image_array = np.array(raw_image.convert('RGB'))
        mask_array = np.array(mask)
        
        box, crop_flag = self.crop_img_mask_squre(image_array, mask_array, max_expand = self.max_expand)
        mask_array = mask_array[box[0]:box[1], box[2]:box[3]]
        image_array = image_array[box[0]:box[1], box[2]:box[3]]
        
        # original size 疑问：作用是？box 需要作为参数么？
        original_width, original_height = image_array.shape[0],image_array.shape[1]
        original_size = torch.tensor([original_height, original_width])

        image = self.transform(Image.fromarray(image_array))  

        # 将掩码应用到原始图像
        masked_image_array = np.copy(image_array)
        masked_image_array[mask_array == 0] = 0

        masked_image = self.transform(Image.fromarray(masked_image_array))

        # 使用sobel算子获得边缘图作为control_cond
        control_cond_image =  self.sobel(image_array.astype(np.float32), mask_array.astype(np.float32))
        control_cond_image = self.transform(Image.fromarray(control_cond_image))

   
        # drop 随机丢弃
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,                              # resize为正方形 -1~+1 1024*1024
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "drop_image_embed": drop_image_embed,        # flag: 标志是否丢弃 img_embeds
            "original_size": original_size,              # ?疑问
            "target_size": torch.tensor([self.size, self.size]),
            #modify:
            "control_cond_image":control_cond_image,
            "masked_image":masked_image
        }
        
    def __len__(self):
        return len(self.data)


# 按照batchsize的维度，堆叠为[bs,...]
def collate_fn(data):
    images = torch.stack([example["image"] for example in data])                           # torch.Size([4, 3, 1024, 1024])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)     # torch.Size([4, 77])
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0) 
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])            # torch.Size([4, 2])
    target_size = torch.stack([example["target_size"] for example in data])                # torch.Size([4, 2])
    control_cond_images = torch.stack([example["control_cond_image"] for example in data]) # torch.Size([4, 3, 1024, 1024])
    masked_images = torch.stack([example["masked_image"] for example in data])             # torch.Size([4, 3, 1024, 1024])
    

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "target_size": target_size,
        "control_cond_images": control_cond_images,
        "masked_images":masked_images
    }
    
 
'''
def sobel(img, mask, thresh = 50, erode_kernel_size = 3,iterations = 2):
    #Calculating the high-frequency map.
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((erode_kernel_size,erode_kernel_size),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = iterations) #2次形态学腐蚀：mask向内收缩
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize) 
    sobel_X = cv2.convertScaleAbs(sobelx) #计算图像在水平方向的梯度
    sobel_Y = cv2.convertScaleAbs(sobely) #计算图像在垂直方向的梯度
    #两个梯度图的绝对值相加，得到图像的梯度强度图
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    # 梯度强度图与掩码相乘，以消除无关区域的干扰
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr

# img  = Image.open("/home/lijiahui/dataset/DAVIS_2016/JPEGImages/480p/bear/00000.jpg")
# mask = Image.open("/home/lijiahui/dataset/DAVIS_2016/Annotations/480p/bear/00000.png")
# image_array = np.array(img)
# mask_array = np.array(mask)

# img_ = sobel(image_array,mask_array, erode_kernel_size = 3, iterations = 2)
# img_PIL = Image.fromarray(img_)
# img_PIL.save("sobel_anydoor_util.png")
# cv2.imwrite("sobel_anydoor_util.png",img_)

'''

tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

dataset = DAVIS_Dataset(tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=1024)
print("total datasets num is :",dataset.__len__())
dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=4)
    
def vis_sample(item):
    images = item['images'] + 1.0   
    masked_images = item['masked_images'] + 1.0    
    control_cond_images = item['control_cond_images'] + 1.0   
    grid_imgs = torch.cat([images, masked_images, control_cond_images],dim=0)
    save_image(grid_imgs,'/home/lijiahui/tmp/instantID_modify/sample_vis_images.png',nrow= 4)
   
print('len dataloader: ', len(dataloader))
for i, data in enumerate(dataloader):  
    vis_sample(data) 
    print("vis done")
    break

        