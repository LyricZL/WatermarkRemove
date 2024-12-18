import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class TransformerBlock(nn.Module):
    def __init__(self, channels, block_size=16):
        super(TransformerBlock, self).__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.attention = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        block_size = self.block_size

        # 确保图像可以整除 block_size
        assert H % block_size == 0 and W % block_size == 0, "Image size must be divisible by block size."

        # 将图像划分为小块
        x_blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)  # Shape: [B, C, num_blocks_h, num_blocks_w, block_size, block_size]
        num_blocks_h, num_blocks_w = x_blocks.shape[2], x_blocks.shape[3]

        # 处理每个块
        outputs = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = x_blocks[:, :, i, j, :, :]  # 单个块，Shape: [B, C, block_size, block_size]
                qkv = self.qkv(block).view(B, 3, C, block_size * block_size)
                q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
                attention_map = self.attention(torch.bmm(q.permute(0, 2, 1), k))  # Shape: [B, block_size*block_size, block_size*block_size]
                out = torch.bmm(attention_map, v.permute(0, 2, 1))
                outputs.append(out.permute(0, 2, 1).view(B, C, block_size, block_size))

        # 合并块
        output = torch.cat([torch.cat(outputs[i*num_blocks_w:(i+1)*num_blocks_w], dim=3) for i in range(num_blocks_h)], dim=2)

        return self.proj(output) + x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.attention = AttentionBlock(channels)
        self.transformer = TransformerBlock(channels)  # 新增 Transformer Block

    def forward(self, x):
        conv_out = self.conv_block(x)
        attention_out = self.attention(conv_out)
        transformer_out = self.transformer(attention_out)  # Transformer Block 输出
        return x + transformer_out

class EnhancedGenerator(nn.Module):
    def __init__(self):
        super(EnhancedGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(12)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)

        # 调整尺寸
        x = pad_to_block_size(x, block_size=16)

        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


def pad_to_block_size(image, block_size=16):
    """
    调整输入图像到能够被 block_size 整除的尺寸。
    """
    _, _, h, w = image.shape
    new_h = (h + block_size - 1) // block_size * block_size
    new_w = (w + block_size - 1) // block_size * block_size
    pad_h = new_h - h
    pad_w = new_w - w
    padding = (0, pad_w, 0, pad_h)  # 左右、上下
    image = F.pad(image, padding, mode="reflect")
    #print(f"Padded image from ({h}, {w}) to ({new_h}, {new_w})")  # 调试信息
    return image