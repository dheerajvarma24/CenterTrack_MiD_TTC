from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

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
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]

            if opt.headtype == 'mlp_mixer':
                # let the patch size be 8X8             
                MlpBlock = nn.Sequential(
                   MixerBlock(num_tokens=8*8, num_channels=last_channel, use_ln=True),
                   MLP(embedding_dim_in=last_channel, hidden_dim=last_channel, embedding_dim_out=classes))
                #out = nn.Conv2d(head_conv[-1], classes, kernel_size=1, stride=1, padding=0, bias=True)
                
                self.__setattr__(head, MlpBlock)
                continue


            if len(head_conv) > 0:
              # RepMixer (reparameterize block) in head type
              if opt.headtype == 'rep_block':
                 # RepMixer
                  '''conv = RepMixer(
                      in_channels=last_channel,
                      out_channels=last_channel,
                      stride=1,
                      groups=last_channel,
                      rbr_conv_kernel_list=[3],
                      use_bn_conv=True,
                      act_layer=nn.ReLU,
                      inference_mode=False,
                      skip_include_bn=False,
                      use_act = False,
                  )'''
                  conv = RepMixerBlock(
                    last_channel,
                    kernel_size=3,
                    mlp_ratio=4.0,
                    act_layer=nn.ReLU,
                    drop=0.0,
                    drop_path=0.0,
                    use_layer_scale=True,
                    layer_scale_init_value=1e-5,
                    inference_mode=False,
                  )

                  out = nn.Conv2d(last_channel, classes, kernel_size=1, stride=1, padding=0, bias=True)                    

              else:
                # Conv block               
                conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
                out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
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
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
            if self.opt.headtype == 'mlp_mixer':
                windows = window_partition(feats[s], 8, channel_last=False)
                z[head] = self.__getattr__(head)(windows)
                z[head]  = window_reverse(z[head], 8, x.shape[3]//self.opt.down_ratio, x.shape[2]//self.opt.down_ratio)
            else:
                z[head] = self.__getattr__(head)(feats[s])

          out.append(z)
      return out
    



# Reference: https://github.com/balala8/FastViT_pytorch/blob/main/model.py
from typing import List, Tuple, Optional
from timm.models.layers import trunc_normal_, DropPath

class RepMixer(nn.Module):
    """
    MobileOne-style residual blocks, including residual joins and re-parameterization convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        rbr_conv_kernel_list: List[int] = [7, 3],
        use_bn_conv: bool = False,   
        act_layer: nn.Module = nn.ReLU,
        skip_include_bn: bool = True,
        use_act: bool = True,
    ) -> None:
        """Construct a Re-parameterization module.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for convolution.
        :param groups: Number of groups for convolution.
        :param inference_mode: Whether to use inference mode.
        :param rbr_conv_kernel_list: List of kernel sizes for re-parameterizable convolutions.
        :param use_bn_conv: Whether the bn is in front of conv, if false, conv is in front of bn
        :param act_layer: Activation layer.
        :param skip_include_bn: Whether to include bn in skip connection.
        """
        super(RepMixer, self).__init__()

        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rbr_conv_kernel_list = sorted(rbr_conv_kernel_list, reverse=True)
        self.num_conv_branches = len(self.rbr_conv_kernel_list)
        self.kernel_size = self.rbr_conv_kernel_list[0]
        self.use_bn_conv = use_bn_conv
        self.skip_include_bn = skip_include_bn

        self.activation = act_layer() if use_act else nn.Identity()          
        

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.kernel_size // 2,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            if out_channels == in_channels and stride == 1:
                if self.skip_include_bn:
                    # Use residual connections that include BN
                    self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)
                else:
                    # Use residual connections
                    self.rbr_skip = nn.Identity()
            else:
                # Use residual connections
                self.rbr_skip = None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for kernel_size in self.rbr_conv_kernel_list:
                if self.use_bn_conv:
                    rbr_conv.append(
                        self._bn_conv(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )
                else:
                    rbr_conv.append(
                        self._conv_bn(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )

            self.rbr_conv = nn.ModuleList(rbr_conv)
    
    
    def _bn_conv(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
      ) -> nn.Sequential:
        """Add bn first, then conv"""
        mod_list = nn.Sequential()
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=in_chans))
        mod_list.add_module(
            "dwconv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )

        # adding additional pointwise convolutions that are not present in the RepMixer paper
        mod_list.add_module(
            "pconv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            ),
        )
        return mod_list
    

    def _conv_bn(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ) -> nn.Sequential:
        """First conv, then bn

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=out_chans))
        return mod_list
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Other branches
        out = identity_out
        for ix in range(self.num_conv_branches):
            out = out + self.rbr_conv[ix](x)
        return self.activation(out)


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.ReLU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.ReLU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.ReLU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        '''self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
            )'''

        self.token_mixer = RepMixer(
            in_channels=dim,
            out_channels=dim,
            stride=1,
            groups=dim,
            rbr_conv_kernel_list=[3],
            use_bn_conv=True,
            act_layer=nn.ReLU,
            inference_mode=False,
            skip_include_bn=False,
            use_act = False,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x
    



#################### MLP-Mixer ####################

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            #nn.GELU(),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, num_channels, use_ln=True):
        super(MixerBlock, self).__init__()
        self.use_ln = use_ln
        if use_ln:
            self.ln_token = nn.LayerNorm(num_channels)
            self.ln_channel = nn.LayerNorm(num_channels)
        self.token_mix = MlpBlock(num_tokens, num_tokens * 2)
        self.channel_mix = MlpBlock(num_channels, num_channels * 2)

    def forward(self, x):
        if self.use_ln:
            out = self.ln_token(x)
        else:
            out = x
        out = out.transpose(-1, -2)
        x = x + self.token_mix(out).transpose(-1, -2)
        if self.use_ln:
            out = self.ln_channel(x)
        else:
            out = x
        x = x + self.channel_mix(out)
        return x

class MLP(nn.Module):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=nn.ReLU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = nn.Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    

def window_partition(x, window_size, channel_last=True):
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

def window_reverse(windows, window_size, W, H):
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