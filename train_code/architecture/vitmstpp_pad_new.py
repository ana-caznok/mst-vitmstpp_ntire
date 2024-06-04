import torch
from torch import nn
from segment_anything import sam_model_registry

try: 
    from .MST_Plus_Plus import MST_Plus_Plus
    from unet import DoubleConv
except: 
   from HyperSkinUtils.baseline.MST_Plus_Plus import MST_Plus_Plus 
   from unet.unet import DoubleConv


class DownsampleBatch():
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, 
                 x: torch.Tensor) -> torch.Tensor:
        return x[:, :, ::self.factor, ::self.factor]
    
    def __str__(self) -> str:
        return f"Downsample by a factor of {self.factor}, assuming 4D batch input"
    

class VITMSTPP_Pad(nn.Module):
    '''
    Deal with the impossibility of working at 1024 in MSTPP with clever use of ViT encoder compression, 
    strided convolutions and unet-like encoding/decoding
    '''
    def __init__(self, mst_size=6, C_input=3, total_channels=31, norm="instance"):
        '''
        mst_size: stage argument for MSTPP
        '''
        super().__init__()

        self.total_channels = total_channels
        self.C_input = C_input

        # Encoder Compression vit, 1024 -> 512
        self.vit = sam_model_registry["vit_b"](checkpoint="sam/sam_vit_b_01ec64.pth").image_encoder
        for param in self.vit.parameters():
            param.requires_grad = False

        # Warn if performing input upsample
        self.upsample_warning = False

        # One extra compression
        # self.compression_convolution = DoubleConv(in_ch=4 + 4, out_ch=4, norm=norm, reduce=True, dim='2d')
        self.vit_shape = [4, 512, 512]
        self.compression_convolution = DoubleConv(in_ch=C_input + self.vit_shape[0], out_ch=self.vit_shape[0], norm=norm, reduce=True, dim='2d')

        # Bottleneck at MSTPP
        # self.mstpp = MST_Plus_Plus(in_channels=4 + 4, out_channels=total_channels, n_feat=total_channels, stage=mst_size)
        self.mstpp = MST_Plus_Plus(in_channels=C_input + self.vit_shape[0], out_channels=self.total_channels, n_feat=self.total_channels, stage=mst_size)

        # Using Gerard SegNet upsample logic in decoding
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="nearest")
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")

        #different upsample in 

        self.downsample_4 = DownsampleBatch(4)
        self.downsample_2 = DownsampleBatch(2)

        self.padding = Padder() if False else nn.Identity()
        
        # Decodes concatenation of all encoder outputs
        self.decoding_convolution = nn.Sequential(DoubleConv(in_ch = self.total_channels + C_input + 2*self.vit_shape[0], out_ch = self.total_channels*2, norm=norm, reduce=False, dim='2d'),
                                                  DoubleConv(in_ch = self.total_channels*2, out_ch = self.total_channels, norm=norm, reduce=False, dim='2d'))
        
        # Linear mapping for final output
        self.out_conv = nn.Conv2d(in_channels=self.total_channels, out_channels=self.total_channels, kernel_size=1, padding=0, stride=1, bias=False)

        print(f"Initialized VITMSTPPUNet with MSTPP stages: {mst_size} and normalization {norm}")
    
    def pad_asim(self,X,Y): 

        if X < Y:
            padding_left = (Y - X) // 2
            padding_right = Y - X - padding_left
            padding = (padding_left, padding_right, 0, 0)
        else:
            padding_top = (X - Y) // 2
            padding_bottom = X - Y - padding_top
            padding = (0, 0, padding_top, padding_bottom)

        return padding 
    
    def crop_back(self,padding, x):

        padding_left, padding_right, padding_top, padding_bottom = padding
        # Calcula as coordenadas de corte
        crop_left = padding_left
        crop_right = x.shape[-1] - padding_right
        crop_top = padding_top
        crop_bottom = x.shape[-2] - padding_bottom
    
        # Realiza o corte
        x = x[:, :, crop_top:crop_bottom, crop_left:crop_right] 
        return x 
    
    def pad_512(self,X,Y): 
        padding_left, padding_right, padding_top, padding_bottom = 0,0,0,0

        if X < 512:
            padding_left = (512 - X) // 2
            padding_right = 512 - X - padding_left
        
        if Y< 512:
            padding_top = (512 - Y) // 2
            padding_bottom = 512 - Y - padding_top
        
        padding = (padding_left, padding_right, padding_top, padding_bottom)

        return padding
    
    def convert2RGB(self, x_input): 
        # Convert to "signal boosted RGB" format
        # Attempts at broadcasting didn't work
        x = x_input[:, :3]
        for c in range(3):
            x[:, c:c+1] = (x[:, c:c+1] + x_input[:, 3:4])/2
        x = x[:, :3]  # remove infrared channel
        return x 
    
    def not_4_div_scale(self, X, Y,  x, x_input, x_half, x_quarter ): 
        print('entrei aqui')
        raise DeprecationWarning("Não quero usar interpolação mais")
        bq, cq, yq, xq = x_quarter.shape

        if yq*4>Y: 
            scf_y = Y/(yq*4)
        else: 
            scf_y = (yq*4)/Y
        
        if xq*4>X: 
            scf_x = X/(xq*4)
        else: 
            scf_x = (xq*4)/X

        print('scale factor:', scf_y, scf_x)
        x_quarter_upsample = torch.nn.functional.interpolate(self.upsample_4(x_quarter), scale_factor=(scf_y, scf_x), mode="nearest")
        
        x_upsample = torch.nn.functional.interpolate(self.upsample_4(x), scale_factor=(scf_y, scf_x), mode="nearest")

        x = torch.cat([x_upsample,  # 3D upsample MSTPP output (31 channels)
                        x_quarter_upsample,  # x256 still has 4 channels, 2D upsample by 4
                    self.upsample_2(x_half),  # x512 still has 4 channels, 2D upsample by 2
                    x_input  # the input
                    ], dim=1)
        del x_quarter_upsample, x_upsample, x_input

        return x

    def forward(self, x_input):

        B, C, Y, X = x_input.shape
        assert C == self.C_input, f"Malformed VITMSTPPUNet input: {x_input.shape}, expected {self.C_input} channels"
        asim = False
        
        #supondo imagem 256x256 -> pad para 512x512 (full), interp para 1024x1024

        if Y != X: #caso X seja diferente de Y, faz um padding para deixa-los iguais 
            asim = True
            print('Imagem assimétrica: y=', Y, ' x=',X, ' asim:', asim)
            padding = self.pad_asim(X,Y)
            x_input = torch.nn.functional.pad(x_input, padding, mode='reflect')
            B, C, Y, X = x_input.shape 
            print('Padding feito: y=',Y, 'x=', X )
        
        if X<512 and X>=256: #caso a imagem seja um patch 482x482 ou 256x256 será feito um padding para deixar a imagem 512x512 e evitar outros dimensionamentos
            asim = True 
            padding = self.pad_512(X,Y)
            x_input = torch.nn.functional.pad(x_input, padding, mode='reflect')
            B, C, Y, X = x_input.shape

        if X<256: #new, para caso a imagem seja menor que 256
            asim = True 
            padding_256 = ((256-X)//2, (256-X)//2, (256-Y)//2, (256-Y)//2)
            x_input = torch.nn.functional.pad(x_input, padding_256, mode='reflect')
            B, C, Y, X = x_input.shape

            padding = self.pad_512(X,Y)
            x_input = torch.nn.functional.pad(x_input, padding, mode='reflect')
            B, C, Y, X = x_input.shape
            padding = (padding_256[0] + padding[0], 
                       padding_256[1] + padding[1], 
                       padding_256[2] + padding[2], 
                       padding_256[3] + padding[3])
                        #considerando o padding pra 256 e depois o para 512 

        # Convert to "signal boosted RGB" format
    
        # If not 1024 input, interpolate in GPU
        scale_Y = 1024/Y
        scale_X = 1024/X

        if X != 1024:
            x = torch.nn.functional.interpolate(x_input, scale_factor=(scale_Y, scale_X), mode="nearest")
            if not self.upsample_warning:
                print("WARNING: VITMSTPPUNET performing interpolation to ViT required 1024 spatial resolution.")
                self.upsample_warning = True
            if self.C_input != 3: 
                x = self.convert2RGB(x)
        else: 
           x = x_input

        # ViT encoder makes 3x1024x1024 -> 256x64x64 -> 4, 512, 512 image
        x_half = self.vit(x).reshape(B, 4, 512, 512)  # poderia referenciar o self.vit_shape aqui

        # In case input was not 1024, scale back embedding with inverse factor
        if scale_Y != 1 or scale_X != 1:
            x_half = torch.nn.functional.interpolate(x_half, scale_factor=(1/scale_Y, 1/scale_X), mode="nearest")

        # Compress further with convolution from (3+4) 7x512x512 to 4x256x256 (7 channelsin, 4 channels out)
        x_quarter = self.compression_convolution(torch.cat([x_half, self.downsample_2(x_input)], dim=1))  # 3 channels from x_input + 4 channels from vit embedding

        # MSTPP works with 256x256, expands channels. x_quarter = 4x256x256 concatenate with 3x256x256 (downsample_4 output)
        x = self.mstpp(torch.cat([x_quarter, self.downsample_4(x_input)], dim=1))

        if Y!=x.shape[2]*4 or X!=x.shape[3]*4: #aceita tamanhos de imagens que não são divisíveis por 4 

            x = self.not_4_div_scale(X,Y,x,x_input,x_half,x_quarter)
    
        else: 
            x = torch.cat([self.upsample_4(x),  # 3D upsample MSTPP output (31 channels)
                        self.upsample_4(x_quarter),  # x256 still has 4 channels, 2D upsample by 4
                        self.upsample_2(x_half),  # x512 still has 4 channels, 2D upsample by 2
                        x_input  # the input
                        ], dim=1)
        del x_quarter
        del x_half
        del x_input #new

        # Decode from the concatenated upsampled features and input
        x = self.decoding_convolution(x)
        x = self.out_conv(x)

        if asim: 
            x = self.crop_back(padding,x)
           

        return torch.clip(x, 0, 1)


if __name__ == "__main__":
    import torchinfo 

    vitmstpp = VITMSTPP_Pad()
    # vitmstpp.cuda()

    torchinfo.summary(vitmstpp, input_size=(1, 3, 1024, 1024), device=torch.device("gpu"))
    torchinfo.summary(vitmstpp, input_size=(1, 3, 128, 128), device=torch.device("gpu"))
