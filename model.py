import torch 
from torch import nn
from diffusers import UNet2DModel
from ddim_sche import DDIMSche


class DDPM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(**config.model_config)
        self.sche = DDIMSche(**config.sche_config)
        self.loss_fn = nn.MSELoss()
  
        
    def forward(self, noisy_image, t):
        return self.model(noisy_image, t).sample
    
    
    def train_loss(self, batch):
        imgs = batch
        b_s = imgs.shape[0]
        # add noise
        t = torch.randint(low=0, high=self.sche.num_train_steps, size=(b_s, ), device=imgs.device)
        out = self.sche.add_noise(imgs, t)

        noisy_img, noise = out['noisy_img'], out['noise'] 
        pred_noise = self.model(noisy_img, t).sample
        train_loss = self.loss_fn(pred_noise, noise)
        return train_loss
    
    @torch.no_grad()
    def sample(self, b_s):
        self.eval()
        noise = torch.randn((b_s, 
                             self.config.model_config.in_channels, 
                             self.config.model_config.sample_size, 
                             self.config.model_config.sample_size), device=self.config.device)
        imgs = self.sche.ddim_sample(self.model, noise)
        return imgs
        