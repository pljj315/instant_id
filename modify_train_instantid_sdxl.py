import os
os.environ['HF_HOME'] = '/mnt/nfs/file_server/public/xy/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import multiprocessing

import random
import argparse
from pathlib import Path
import itertools
import time
import cv2

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection


from ip_adapter.resampler import Resampler    
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from modify_instantID import FrozenDinoV2Encoder

    
# Dataset
class DAVIS_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, tokenizer_2, size=1024, max_expand = True,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, 
                 image_root_path="/home/lijiahui/dataset/DAVIS_2016",
                 pairs_file_path="/home/lijiahui/dataset/DAVIS_2016/ImageSets/1080p/train.txt"):
        super().__init__()

        
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.max_expand = max_expand
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.transform = transforms.Compose([
            transforms.Resize((self.size,self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                #       0~1
            transforms.Normalize([0.5], [0.5]),]) #归一化 -1~1
        self.transform_2 = transforms.Compose([
            transforms.Resize((self.size//14 * 14, self.size//14*14), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                #       0~1
            transforms.Normalize([0.5], [0.5]),]) #归一化 -1~1
        # list of dict: [{"image_path": "/JPEGImages/1080p/bear/00000.jpg", "mask_path": "/Annotations/1080p/bear/00000.png", "text": "A dog"}]
        data = []
        with open(pairs_file_path, "r") as lines:
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
        # print(box,box_height,box_width)

        if box_width == box_height:
            print("本身mask就是squre")
            return (y1,y2,x1,x2),crop_flag
        
        side_length = max(box_width, box_height)

        # crop:
        if side_length > min(H, W):
            # print("mask 形状不适合直接 to square,需要中心裁剪")
            side_length = min(H,W)
            crop_side = 'x' if box_width > box_height else 'y'  # 哪边超长就裁剪哪边
            if crop_side =="x":
                x1= x1 + (box_width - side_length)//2
                x2= x1 + side_length
            else:
                y1 = y1 + (box_height - side_length)//2
                y2 = y1 + side_length
            crop_flag = 1
            box_height ,box_width = y2-y1, x2-x1

        # expand:
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
                y2= y1 + side_length
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
        # 输出前的检查
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
        
        image_array = np.array(raw_image)
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

        dinov2_image = self.transform_2(Image.fromarray(masked_image_array))

        # 使用sobel算子获得边缘图作为control_cond
        control_cond_image =  self.sobel(image_array.astype(np.float32), mask_array.astype(np.float32))
        control_cond_image = self.transform(Image.fromarray(control_cond_image).convert("RGB"))

   
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
            "yyxx": torch.tensor(box),
            #modify:
            "control_cond_image":control_cond_image,
            "masked_image":masked_image,
            "dinov2_image":dinov2_image
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
    yyxx = torch.stack([example["yyxx"] for example in data])                              # torch.Size([4, 4])
    control_cond_images = torch.stack([example["control_cond_image"] for example in data]) # torch.Size([4, 3, 1024, 1024])
    masked_images = torch.stack([example["masked_image"] for example in data])             # torch.Size([4, 3, 1024, 1024])
    dinov2_images = torch.stack([example["dinov2_image"] for example in data])             # [4, 3, 1022, 1022]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "yyxx": yyxx,
        "control_cond_images": control_cond_images,
        "masked_images":masked_images,
        "dinov2_images":dinov2_images
    }
    
class InstantID_SDXL(torch.nn.Module):
    def __init__(self, unet, controlnet, image_proj_model, adapter_modules, instanid_ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet               # instantID中的：Identity Model
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules


        if instanid_ckpt_path is not None:
            self.load_from_checkpoint(instanid_ckpt_path)

    def forward(self, 
                noisy_latents,   # [4, 4, 128, 128]
                timesteps,       # [4]
                text_embeds,     # [4, 77, 2048]
                image_embeds,    # [4, 5330, 1536]
                controlnet_image, 
                text_time_added_cond_kwargs ):  # unet的added_cond_kwargs与controlnet的added_cond_kwargs  
    
        ip_tokens = self.image_proj_model(image_embeds) # [4, 5330, 2048]

        # 先cat，后续在 anntnprocessoe模块中，会进行划分：encoder_hidden_states, ip_hidden_states
        encoder_hidden_states = torch.cat([text_embeds, ip_tokens], dim=1) # [4, 5330+77, 2048]

        # controlnet 获得res_samples
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents, 
            timesteps,
            encoder_hidden_states=text_embeds,  # 使用 face_embeddings? # [4, 5330, 2048]
            controlnet_cond=controlnet_image, # 高频图  
            added_cond_kwargs=text_time_added_cond_kwargs, # time_ids,text_embeds 时间和文字
            return_dict=False, )

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, 
                               timesteps, 
                               encoder_hidden_states, 
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample,
                               added_cond_kwargs=ip_tokens  # 可以修改：可以不使用added_cond_kwargs，# 此处仿照ip_adapter_embeds
                               ).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_control_sum = torch.sum(torch.stack([torch.sum(p) for p in self.controlnet.parameters()]))
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict
        self.controlnet.load_state_dict(state_dict["controlnet"], strict=True)
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_control_sum = torch.sum(torch.stack([torch.sum(p) for p in self.controlnet.parameters()]))
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_control_sum != new_control_sum, "Weights of controlnet_model did not change!"
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--pretrained_model_name_or_path",type=str,  
        default="stabilityai/stable-diffusion-xl-base-1.0", help="Path to pretrained model or model identifier from huggingface.co/models.",)
    
    parser.add_argument("--pretrained_instantid_path",type=str,      
        default=None, help="Path to pretrained instanID model. If not specified weights are initialized randomly.",)
    
    parser.add_argument("--data_root_path",type=str,                 
        default="/home/lijiahui/dataset/DAVIS_2016",  help="Training data root path",)
    
    parser.add_argument("--pairs_file_path",type=str,                 
        default="/home/lijiahui/dataset/DAVIS_2016/ImageSets/1080p/train.txt",  help="Training data root path",)
    
    parser.add_argument("--output_dir",type=str,
        default="/home/lijiahui/tmp/InstantID/output",help="The output directory where the model predictions and checkpoints will be written.",)
    
    parser.add_argument("--logging_dir",type=str,
        default="logs",help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
    
    parser.add_argument("--resolution",type=int,        default=1024, help=("The resolution for input images"),)
    parser.add_argument("--learning_rate", type=float,  default=1e-5, help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float,   default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100   )
    parser.add_argument("--train_batch_size", type=int, default=1,    help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--noise_offset", type=float,   default=None, help="noise offset")
    parser.add_argument("--dataloader_num_workers",type=int, default=2,    help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
    parser.add_argument("--save_steps",type=int,             default=2000, help=( "Save a checkpoint of the training state every X updates"),)
    parser.add_argument("--mixed_precision",type=str,        default="fp16", choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--report_to",type=str,              default="tensorboard", 
            help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    print("logging_dir", logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            print("output dir 检查完毕")

    # 加载必要的预训练组件：
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(      args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(   args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(    args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(            args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(    args.pretrained_model_name_or_path, subfolder="unet")
    
    # 从unet初始化controlnet
    controlnet = ControlNetModel.from_unet(unet)
    # 加载dinov2 预训练模型(freeze已经冻结): dinov2_vitg14=> 1536
    image_encoder = FrozenDinoV2Encoder(DINOv2_weight_path="/home/lijiahui/models/model-dinov2/dinov2_vitg14_pretrain.pth" )
    image_encoder.requires_grad_(False)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    weight_dtype = torch.float32    # 默认使用 fp32
    print('accelerator.mixed_precision:',accelerator.mixed_precision)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet.to(accelerator.device, dtype=weight_dtype) 
    vae.to(accelerator.device)  # vae的精度必须是fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    
    # 投影 resampler: 效果比MLP好
    print('unet.config.cross_attention_dim:',unet.config.cross_attention_dim, # 2048
          'unet.config.block_out_channels:',unet.config.block_out_channels)   # [320, 640, 1280] 

    # 函数set_ip_adapter():init adapter modules: 修改unet：把ip adapter挂在unet上
    unet_sd = unet.state_dict()
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim ## 2048
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor().to(accelerator.device, dtype=weight_dtype)
        else:            
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size, 
                cross_attention_dim=cross_attention_dim, 
                scale= 1 ,
                num_tokens=4).to(accelerator.device, dtype=weight_dtype)
            # 权重的初始化：
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],}
            attn_procs[name].load_state_dict(weights)     
    unet.set_attn_processor(attn_procs)

    # 函数set_ip_adapter_scale() 注意ip_attn_processor的scales可以修改
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values()).to(accelerator.device, dtype=weight_dtype) 
    
    image_proj_model = torch.nn.Linear(1536,2048).to(accelerator.device, dtype=weight_dtype) 
    # # 函数set_image_proj_model()
    # image_proj_model = Resampler(
    #     dim=768, 
    #     depth=4, 
    #     dim_head=64, 
    #     heads=20,
    #     num_queries=proj_num_tokens,  #16
    #     embedding_dim= 1536,    #proj_image_emb_dim,    #512           # 输入的维度
    #     output_dim=  2048  , #unet.config.cross_attention_dim,                 # 输出的维度应该等于cross_attn_dim=2048
    #     ff_mult=4,).to(accelerator.device, dtype=weight_dtype)


    controlnet.train() 
    image_proj_model.train()
    adapter_modules.train()

    instantID = InstantID_SDXL(unet, controlnet, image_proj_model, adapter_modules, args.pretrained_instantid_path)
    
    # optimizer 
    params_to_opt = itertools.chain(instantID.controlnet.parameters(),  instantID.image_proj_model.parameters(),  instantID.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader 
    train_dataset = DAVIS_Dataset(tokenizer=tokenizer, tokenizer_2=tokenizer_2,size=args.resolution,
                            image_root_path=args.data_root_path, pairs_file_path=args.pairs_file_path )
    print("total datasets num is :",train_dataset.__len__())
    train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers, )
    
    # Prepare everything with our `accelerator`.
    instantID, optimizer, train_dataloader,image_encoder = accelerator.prepare(instantID, optimizer, train_dataloader,image_encoder)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(instantID):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # 前向加噪/前向传播过程 (this is the forward diffusion process) 
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

#################################################################################
                with torch.no_grad():
                    # 将图像PIL传递给dinov2_vits14模型获取特征
                    dinov2_image_embeds = image_encoder(batch["dinov2_images"]).to(accelerator.device, dtype=weight_dtype)
                
                image_embeds_ = []
                for dinov2_image_embed, drop_image_embed in zip(dinov2_image_embeds, batch["drop_image_embeds"]): 
                    # print(dinov2_image_embed.shape) #torch.Size([ 5330, 1024])
                    if drop_image_embed == 1:  # 是否丢弃img_embed的标志flag
                        image_embeds_.append(torch.zeros_like(dinov2_image_embed))
                    else:
                        image_embeds_.append(dinov2_image_embed)
                image_embeds = torch.stack(image_embeds_) #将若干个张量在维度上连接,生成一个'扩维'的张量

#################################################################################

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]     # [4, 77, 768]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    text_embeds_2 = encoder_output_2.hidden_states[-2] # [4, 77, 1280]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) #[4, 77, 2048]
                    
                    pooled_text_embeds = encoder_output_2[0]           # [4, 1280]

                
                # modify: ControlNet conditioning.高频图
                controlnet_image = batch["control_cond_images"].to(accelerator.device)

                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["yyxx"].to(accelerator.device),]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)

                #add_time_ids.shape=[4, 6]                            # [4, 1280 + 256*6 = 2816]  
                text_time_added_cond_kwargs = {"text_embeds": pooled_text_embeds,"time_ids": add_time_ids} #controlnet默认配置为'text-time'
                

                noise_pred = instantID(noisy_latents, 
                                       timesteps, 
                                       text_embeds, 
                                       image_embeds, 
                                       controlnet_image, 
                                       text_time_added_cond_kwargs )
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, load_data_time: {}, whole_step_time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
