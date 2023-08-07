from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from .mlp_mixer import MLP, MixerBlock

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.opt = opt
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if self.opt.headtype == 'mlp_mixer':
               # let the patch size be 8X8
               mixer_block = MixerBlock(8*8, last_channel, use_ln=True)
               mlp = MLP(embedding_dim_in=last_channel, hidden_dim=16, embedding_dim_out=classes)
               fc = nn.Sequential(mixer_block, mlp)
            
            elif not self.opt.headtype == 'mlp_mixer':
              head_conv = head_convs[head]
              if len(head_conv) > 0:
                out = nn.Conv2d(head_conv[-1], classes, 
                      kernel_size=1, stride=1, padding=0, bias=True)
                conv = nn.Conv2d(last_channel, head_conv[0],
                                kernel_size=head_kernel, 
                                padding=head_kernel // 2, bias=True)
                convs = [conv]
                for k in range(1, len(head_conv)):
                    convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                                kernel_size=1, bias=True))
                if len(convs) == 1:
                  fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                elif len(convs) == 2:
                  fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), out)
                elif len(convs) == 3:
                  fc = nn.Sequential(
                      convs[0], nn.ReLU(inplace=True), 
                      convs[1], nn.ReLU(inplace=True), 
                      convs[2], nn.ReLU(inplace=True), out)
                elif len(convs) == 4:
                  fc = nn.Sequential(
                      convs[0], nn.ReLU(inplace=True), 
                      convs[1], nn.ReLU(inplace=True), 
                      convs[2], nn.ReLU(inplace=True), 
                      convs[3], nn.ReLU(inplace=True), out)
                if 'hm' in head:
                  fc[-1].bias.data.fill_(opt.prior_bias)
                else:
                  fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def window_partition(self, x, window_size, channel_last=True):
      """
      Args:
          x: (B, W, H, C)
          window_size (int): window size
      Returns:
          windows: (B, num_windows, window_size * window_size, C)
          :param channel_last: if channel is last dim
      """
      if not channel_last:
          #x = x.permute(0, 2, 3 1)
          x = x.permute(0, 3, 2, 1)
      B, W, H, C = x.shape
      x = x.view(B, W // window_size, window_size, H // window_size, window_size, C)
      windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size * window_size, C)
      return windows
    
    def window_reverse(self, windows, window_size, W, H):
      """
      Args:
          windows: (B, num_windows, window_size*window_size, C)
          window_size (int): Window size
          W (int): Width of image
          H (int): Height of image
      Returns:
          x: (B, C, W, H)
      """
      B = windows.shape[0]
      x = windows.reshape(B, W // window_size, H // window_size, window_size, window_size, -1)
      #x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, W, H)
      x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
      return x

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.headtype == 'mlp_mixer':
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              windows = self.window_partition(feats[s], 8, channel_last=False)
              # B, num_windows, PXP, C = windows.shape
              mlp_mixer_out = self.__getattr__(head)(windows)          
              z[head]  = self.window_reverse(mlp_mixer_out, 8, x.shape[3]//self.opt.down_ratio, x.shape[2]//self.opt.down_ratio)
          out.append(z)
      elif self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
