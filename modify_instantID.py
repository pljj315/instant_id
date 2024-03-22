import torch
from ip_adapter.resampler import Resampler,ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from ip_adapter.attention_processor import region_control

from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# ——————————————————————————————————modify DINOV2特征提取/embed:————————————
import sys
sys.path.append("./dinov2")
import hubconf
import torch.nn as nn

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenDinoV2Encoder(AbstractEncoder): # 疑问 待修改
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, DINOv2_weight_path, device='cuda', freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitg14()             # 模型结构
        state_dict = torch.load(DINOv2_weight_path)  # 模型权重
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()

        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        image = (image.to(self.device)  - self.image_mean.to(self.device)).unsqueeze(0) / (self.image_std.to(self.device))
        image = image.squeeze(0)
        print(image.shape)
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]      # [1, 5329, 1536]
        image_features  = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1) # [1, 1, 1536]
        # 图像特征和其他特征（tokens）拼接在一起
        hint = torch.cat([image_features,tokens],1)  # [1, 5330, 1536]
        return hint

    def encode(self, image): # 与forward方法相同
        return self(image)

#————————————————————————————————————modify proj/Resampler 投影:————————————


def load_ip_adapter_instantid(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=0.5):     
    self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
    self.set_ip_adapter(model_ckpt, num_tokens, scale)
    
def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16): 
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_emb_dim,
        output_dim=self.unet.config.cross_attention_dim,
        ff_mult=4,
    )

    
    self.image_proj_model = image_proj_model.to(self.device, dtype=self.dtype)
    state_dict = torch.load(model_ckpt, map_location="cpu")
    if 'image_proj' in state_dict:
        state_dict = state_dict["image_proj"]
    self.image_proj_model.load_state_dict(state_dict)
    
    self.image_proj_model_in_features = image_emb_dim

def set_ip_adapter(self, model_ckpt, num_tokens, scale):
    
    unet = self.unet
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
        else:
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                cross_attention_dim=cross_attention_dim, 
                                                scale=scale,
                                                num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
    unet.set_attn_processor(attn_procs)
    
    state_dict = torch.load(model_ckpt, map_location="cpu")
    ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
    if 'ip_adapter' in state_dict:
        state_dict = state_dict['ip_adapter']
    ip_layers.load_state_dict(state_dict)

def set_ip_adapter_scale(self, scale):
    unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
    for attn_processor in unet.attn_processors.values():
        if isinstance(attn_processor, IPAttnProcessor):
            attn_processor.scale = scale

def _encode_prompt_image_emb(self, prompt_image_emb, device, num_images_per_prompt, dtype, do_classifier_free_guidance):
    
    if isinstance(prompt_image_emb, torch.Tensor):
        prompt_image_emb = prompt_image_emb.clone().detach()
    else:
        prompt_image_emb = torch.tensor(prompt_image_emb)
        
    prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])
    
    if do_classifier_free_guidance:
        prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
    else:
        prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
    
    prompt_image_emb = prompt_image_emb.to(device=self.image_proj_model.latents.device, 
                                            dtype=self.image_proj_model.latents.dtype)
    prompt_image_emb = self.image_proj_model(prompt_image_emb)

    bs_embed, seq_len, _ = prompt_image_emb.shape
    prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
    prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
    return prompt_image_emb.to(device=device, dtype=dtype)
