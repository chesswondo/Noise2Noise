import torch
import torch.nn as nn

from .components import ResBlock, AttentionGate

# Decoder Block with PixelShuffle
class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up_conv = nn.Conv2d(in_c, in_c * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.att = AttentionGate(F_g=in_c, F_l=skip_c, F_int=in_c // 2)
        self.conv = ResBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = self.pixel_shuffle(x)
        
        skip = self.att(g=x, x=skip)
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ResBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class AdvancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.b = ResBlock(512, 1024)
        
        self.d1 = DecoderBlock(in_c=1024, skip_c=512, out_c=512)
        self.d2 = DecoderBlock(in_c=512,  skip_c=256, out_c=256)
        self.d3 = DecoderBlock(in_c=256,  skip_c=128, out_c=128)
        self.d4 = DecoderBlock(in_c=128,  skip_c=64,  out_c=64)
        
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        # Bottleneck
        b = self.b(p4)
        
        # Decoder path
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        noise_pred = self.output(d4)
        clean_pred = x - noise_pred
        
        return clean_pred
