from typing import Tuple, List, Optional
from itertools import chain
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass


class ResNet1d(nn.Module):
    """
    一维残差网络（ResNet1d）模块，用于处理一维数据（如音频信号）。

    Args:
        n_channels (int): 输入和输出的通道数。
        kernel_size (int, optional): 卷积核的大小。默认为 7。
        padding (str, optional): 填充方式，可选 'valid' 或 'same'。默认为 'valid'。
        dilation (int, optional): 卷积的膨胀率。默认为 1。
    """
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = 'valid',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        # 卷积核的大小
        self.kernel_size = kernel_size
        # 填充方式
        self.padding = padding
        # 卷积的膨胀率
        self.dilation = dilation
        # 计算填充大小
        self._padding_size = (kernel_size // 2) * dilation

        # 定义第一个一维卷积层
        self.conv0 = nn.Conv1d(
            n_channels,  # 输入通道数
            n_channels,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            padding=padding,  # 填充方式 
            dilation=dilation)  # 膨胀率
        
        # 定义第二个一维卷积层
        self.conv1 = nn.Conv1d(
            in_channels=n_channels,  # 输入通道数
            out_channels=n_channels,  # 输出通道数
            kernel_size=1  # 卷积核大小为 1
        )

    def forward(self, input):
        """
        前向传播方法，实现残差连接。

        Args:
            input (torch.Tensor): 输入张量，形状为 (batch_size, n_channels, length)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, n_channels, length)。
        """
        # 保留输入张量，用于残差连接
        y = input
        # 第一个卷积层
        x = self.conv0(input)
        x = F.elu(x)
        # 第二个卷积层
        x = self.conv1(x)
        if self.padding == 'valid':
            # 如果填充方式为 'valid'，则移除填充部分
            y = y[:, :, self._padding_size:-self._padding_size]
        # 残差连接
        x += y
        x = F.elu(x)
        return x


class ResNet2d(nn.Module):
    """
    二维残差网络（ResNet）模块。

    Args:
        n_channels (int): 输入和输出的通道数。
        factor (int): 通道数的缩放因子，用于控制残差分支的通道数。
        stride (Tuple[int, int]): 卷积的步幅，用于控制特征图的空间分辨率。
    """
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        # The original paper uses layer normalization, but here
        # we use batch normalization.
        super().__init__()

        # 第一个卷积层：3x3 卷积，步幅为1，填充方式为 'same'
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding='same')
        
        # 第一个批归一化层
        self.bn0 = nn.BatchNorm2d(
            n_channels
        )

        # 第二个卷积层：使用 (stride[0] + 2) x (stride[1] + 2) 的卷积核，步幅为 stride
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        
        # 第二个批归一化层
        self.bn1 = nn.BatchNorm2d(
            factor * n_channels
        )

        # 第三个卷积层：1x1 卷积，步幅为 stride
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        
        # 第三个批归一化层
        self.bn2 = nn.BatchNorm2d(
            factor * n_channels
        )

        # 反射填充层，用于匹配特征图的空间分辨率
        self.pad = nn.ReflectionPad2d([
            (stride[1] + 1) // 2,
            (stride[1] + 2) // 2,
            (stride[0] + 1) // 2,
            (stride[0] + 2) // 2,
        ])

        # LeakyReLU 激活函数
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        """
        前向传播方法。

        Args:
            input (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 第一个卷积层和批归一化层
        x = self.conv0(input)
        x = self.bn0(x)
        # 应用激活函数
        x = self.activation(x)
        # 反射填充
        x = self.pad(x)
        # 第二个卷积层和批归一化层
        x = self.conv1(x)
        x = self.bn1(x)

        # 捷径分支（shortcut）
        y = self.conv2(input)
        y = self.bn2(y)

        # 残差连接
        x += y
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    """
    编码器块模块，用于下采样和特征提取。

    Args:
        n_channels (int): 输入和输出的通道数。
        padding (str): 填充方式，'valid' 或 'same'。
        stride (int): 卷积的步幅，用于控制下采样率。
    """
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']

        # 定义编码器层的顺序
        self.layers = nn.Sequential(
            # 第一个一维残差网络模块，通道数减半，膨胀率为1
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            # 第二个一维残差网络模块，通道数减半，膨胀率为3
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            # 第三个一维残差网络模块，通道数减半，膨胀率为9
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            # 一维卷积层，通道数恢复为 n_channels，卷积核大小为 2 * stride，步幅为 stride
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            # ELU 激活函数
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            input (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 通过序列化的层进行前向传播
        return self.layers(input)


class DecoderBlock(nn.Module):
    """
    解码器块模块，用于上采样和特征恢复。

    Args:
        n_channels (int): 输入和输出的通道数。
        padding (str): 填充方式，'valid' 或 'same'。
        stride (int): 转置卷积的步幅，用于控制上采样率。
    """
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']

        # 定义解码器层的顺序
        self.layers = nn.Sequential(
            # 转置卷积层，通道数减半，卷积核大小为 2 * stride，步幅为 stride
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            # ELU 激活函数
            nn.ELU(),
            # 第一个一维残差网络模块，通道数减半，膨胀率为1
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            # 第二个一维残差网络模块，通道数减半，膨胀率为3
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            # 第三个一维残差网络模块，通道数减半，膨胀率为9
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            input (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 通过序列化的层进行前向传播
        return self.layers(input)


class Encoder(nn.Module):
    """
    编码器模块，用于将输入信号压缩成潜在表示。

    Args:
        n_channels (int): 初始卷积层的输出通道数。
        padding (str): 卷积层的填充方式，'valid' 表示不填充，'same' 表示填充使得输出与输入尺寸相同。
    """
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']

        # 定义编码器的层序列
        self.layers = nn.Sequential(
            # 第一个卷积层：1个输入通道，n_channels 个输出通道，卷积核大小为7，填充方式为 padding
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            # ELU 激活函数
            nn.ELU(),
            # 第一个编码块：输入通道数为 2 * n_channels，填充方式为 padding，步幅为 2
            EncoderBlock(2 * n_channels, padding=padding, stride=2),
            # 第二个编码块：输入通道数为 4 * n_channels，填充方式为 padding，步幅为 4
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            # 第三个编码块：输入通道数为 8 * n_channels，填充方式为 padding，步幅为 5
            EncoderBlock(8 * n_channels, padding=padding, stride=5),
            # 第四个编码块：输入通道数为 16 * n_channels，填充方式为 padding，步幅为 8
            EncoderBlock(16 * n_channels, padding=padding, stride=8),
            # 最后一个卷积层：输入通道数为 16 * n_channels，输出通道数为 16 * n_channels，卷积核大小为3，填充方式为 padding
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=3, padding=padding),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            input (torch.Tensor): 输入张量，形状应为 (batch_size, 1, input_length)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, 16 * n_channels, output_length)。
        """
        return self.layers(input)


class Decoder(nn.Module):
    """
    解码器模块，用于将潜在表示重构为原始信号。

    Args:
        n_channels (int): 初始卷积层的输出通道数。
        padding (str): 卷积层的填充方式，'valid' 表示不填充，'same' 表示填充使得输出与输入尺寸相同。
    """
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']

        # 定义解码器的层序列
        self.layers = nn.Sequential(
            # 第一个卷积层：输入通道数为 16 * n_channels，输出通道数为 16 * n_channels，卷积核大小为7，填充方式为 padding
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
            # ELU 激活函数
            nn.ELU(),
            # 第一个解码块：输入通道数为 16 * n_channels，填充方式为 padding，步幅为 8
            DecoderBlock(16 * n_channels, padding=padding, stride=8),
            # 第二个解码块：输入通道数为 8 * n_channels，填充方式为 padding，步幅为 5
            DecoderBlock(8 * n_channels, padding=padding, stride=5),
            # 第三个解码块：输入通道数为 4 * n_channels，填充方式为 padding，步幅为 4
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            # 第四个解码块：输入通道数为 2 * n_channels，填充方式为 padding，步幅为 2
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            # 最后一个卷积层：输入通道数为 n_channels，输出通道数为 1，卷积核大小为7，填充方式为 padding
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            # Tanh 激活函数，用于将输出限制在 [-1, 1] 范围内
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            input (torch.Tensor): 输入张量，形状应为 (batch_size, 16 * n_channels, input_length)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, 1, output_length)。
        """
        return self.layers(input)


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器（Residual Vector Quantizer）
    
    该模块用于将输入的嵌入向量通过多个向量量化器（VQ）进行量化，每个量化器负责对量化残差进行进一步量化。
    通过这种方式，可以在保持较低计算复杂度的同时，实现较高的编码效率。
    
    Args:
        num_quantizers (int): 向量量化器的数量。
        num_embeddings (int): 每个量化器的嵌入向量数量。
        embedding_dim (int): 嵌入向量的维度。
        decay (float, optional): 运行均值更新的衰减因子，默认为0.99。
        code_replace_threshold (float, optional): 替换代码向量的阈值，默认为0.0001。
        eps (float, optional): 用于数值稳定的微小常数，默认为1e-10。
    """
    weight: torch.Tensor  # 嵌入向量权重
    running_mean: torch.Tensor  # 运行均值
    code_count: torch.Tensor  # 代码计数

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        # 向量量化器的数量
        self.num_quantizers = num_quantizers
        # 每个量化器的嵌入向量数量
        self.num_embeddings = num_embeddings
        # 嵌入向量的维度
        self.embedding_dim = embedding_dim
        # 注册缓冲区，用于存储嵌入向量权重、运行均值和代码计数
        self.register_buffer("weight", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("running_mean", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("code_count", torch.empty(num_quantizers, num_embeddings))
        # 运行均值更新的衰减因子
        self.decay = decay
        # 用于数值稳定的微小常数
        self.eps = eps
        # 替换代码向量的阈值
        self.code_replace_threshold = code_replace_threshold
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化参数。
        嵌入向量权重使用均匀分布初始化，运行均值设置为嵌入向量权重，代码计数初始化为1。
        """
        init.uniform_(self.weight)  # 使用均匀分布初始化嵌入向量权重
        self.running_mean[:] = self.weight  # 运行均值设置为嵌入向量权重
        init.ones_(self.code_count)  # 代码计数初始化为1

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        前向传播方法，对输入进行残差向量量化。
        
        Args:
            input (torch.Tensor): 输入张量，形状为 [..., channel]。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 返回量化后的输入、量化索引和承诺损失。
        """
        # input: [..., chennel]
        if self.training:
            # Enabling bitrate scalability with quantizer dropout
            # 在训练模式下启用比特率可伸缩性，通过随机选择量化器数量
            n = random.randrange(1, self.num_quantizers)
        else:
            n = self.num_quantizers
        # 存储量化索引
        codes = []
        # 将输入转换为与运行均值相同的类型，并停止梯度传播
        r = input.type_as(self.running_mean).detach()
        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]
                # r: [..., num_embeddings]
                # 计算输入与当前量化器嵌入向量之间的距离
                dist = torch.cdist(r, w)
                # 选择距离最小的嵌入向量索引
                k = torch.argmin(dist, axis=-1)
                # 存储索引
                codes.append(k)
                # 更新运行均值和代码计数
                self._update_averages(i, r, k)
                # 计算量化残差
                r = r - F.embedding(k, w)
        # 计算量化后的输入
        quantized = input - r
        # 计算承诺损失
        commitment_loss = torch.mean(torch.square(input - quantized.detach()))
        # 更新嵌入向量权重，使用运行均值除以代码计数加eps
        self.weight.data[:] = self.running_mean / torch.unsqueeze(self.eps + self.code_count, axis=-1)
        return quantized, torch.stack(codes, input.ndim - 1), commitment_loss

    def dequantize(self, input: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        """
        对量化后的输入进行反量化。
        
        Args:
            input (torch.Tensor): 量化后的输入张量，形状为 [batch_size, length, num_quantizers]。
            n (Optional[int]): 要反量化的量化器数量，默认为输入的最后一个维度。
        
        Returns:
            torch.Tensor: 反量化后的输出张量。
        """
        # input: [batch_size, length, num_quantizers]
        if n is None:
            n = input.shape[-1]
        assert 0 < n <= self.num_quantizers
        # 初始化反量化结果
        res = 0
        with torch.no_grad():
            for i in range(n):
                k = input[:, :, i]  # 获取当前量化器的量化索引
                w = self.weight[i]  # 获取当前量化器的嵌入向量权重
                res += F.embedding(k, w)  # 反量化并累加
        return res

    def _update_averages(self, i: int, r: torch.Tensor, k: torch.Tensor) -> None:
        # https://arxiv.org/pdf/1906.00446.pdf
        # Generating Diverse High-Fidelity Images with VQ-VAE-2
        # 2.1 Vector Quantized Variational AutoEncode
        """
        更新运行均值和代码计数。
        
        Args:
            i (int): 当前量化器的索引。
            r (torch.Tensor): 量化残差，形状为 [..., embedding_dim]。
            k (torch.Tensor): 量化索引，形状为 [...]。
        """
        # k: [...]
        # 生成 one-hot 编码
        one_hot_k = F.one_hot(torch.flatten(k), self.num_embeddings).type_as(self.code_count)
        # 计算代码计数更新
        code_count_update = torch.mean(one_hot_k, axis=0)
        # 更新代码计数
        self.code_count[i].lerp_(code_count_update, 1 - self.decay)

        # r: [..., embedding_dim]
        # 计算运行均值更新
        r = r.reshape(-1, self.embedding_dim)
        running_mean_update = (one_hot_k.T @ r) / r.shape[0]
        # 更新运行均值
        self.running_mean[i].lerp_(running_mean_update, 1 - self.decay)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def replace_vectors(self) -> int:
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer:

        # The original paper replaces with an input frame randomly
        # sampled within the current batch.
        # Here we replace with random average of running mean instead.
        """
        替换使用不足的代码向量。
        
        Returns:
            int: 被替换的代码向量数量。
        """
        num_replaced = torch.sum(self.code_count < self.code_replace_threshold).item()
        if num_replaced > 0:
            for i in range(self.num_quantizers):
                mask = self.code_count[i] < self.code_replace_threshold
                # mask: [num_quantizers, num_embeddings]
                # 使用随机平均替换使用不足的代码向量
                w = torch.rand_like(self.code_count[i])
                w /= torch.sum(w)
                self.running_mean[i, mask] = w.type_as(self.running_mean) @ self.running_mean[i]
                self.code_count[i, mask] = w.type_as(self.code_count) @ self.code_count[i]

        return num_replaced

    @torch.no_grad()
    def calc_entropy(self) -> float:
        """
        计算代码向量的熵。
        
        Returns:
            float: 代码向量的熵。
        """
        p = self.code_count / (self.eps + torch.sum(self.code_count, axis=-1, keepdim=True))
        return -torch.sum(torch.log(p) * p).item() / self.num_quantizers


class WaveDiscriminator(nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf
    """
    """
    MelGAN 的波形判别器，该判别器用于区分真实波形和生成波形，通过多尺度卷积层来捕捉不同频率范围的声音特征。
    """
    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        """
        初始化 WaveDiscriminator

        Args:
            resolution (int): 下采样倍数。默认为1，表示不进行下采样。
            n_channels (int): 初始卷积层的通道数。默认为4。
        """
        super().__init__()
        assert resolution >= 1
        if resolution == 1:
            # 如果下采样倍数为1，则使用恒等映射，不进行下采样
            self.avg_pool = nn.Identity()
        else:
            # 否则，使用平均池化进行下采样
            self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution)
        # 使用LeakyReLU激活函数
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 定义一系列带权重的卷积层，使用nn.ModuleList存储
        self.layers = nn.ModuleList([
            # 输入通道1，输出通道n_channels，卷积核大小15
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            # 分组卷积，组数4
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            # 分组卷积，组数16
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            # 分组卷积，组数64
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=41, stride=4, padding=20, groups=64)),
            # 分组卷积，组数256
            nn.utils.weight_norm(nn.Conv1d(64 * n_channels, 256 * n_channels, kernel_size=41, stride=4, padding=20, groups=256)),
            # 卷积核大小5
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 256 * n_channels, kernel_size=5, padding=2)),
            # 输出通道1，卷积核大小3
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播方法

        Args:
            x (torch.Tensor): 输入波形张量，形状为 [batch_size, 1, sequence_length]

        Returns:
            List[torch.Tensor]: 包含中间特征图的列表
        """
        # 进行下采样
        x = self.avg_pool(x)
        # 存储中间特征图
        feats = []
        # 遍历除最后一层外的所有层
        for layer in self.layers[:-1]:
            x = layer(x)  # 应用卷积层
            feats.append(x)  # 存储特征图
            x = self.activation(x)  # 应用激活函数
        # 最后一层不应用激活函数，直接存储特征图
        feats.append(self.layers[-1](x))
         # 返回特征图列表
        return feats


class STFTDiscriminator(nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf
    """
    """
    基于STFT的判别器，该判别器使用STFT变换后的频谱图作为输入，通过一系列卷积层来区分真实和生成的音频。
    """
    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256,
        n_channels: int = 32
    ) -> None:
        """
        初始化 STFTDiscriminator

        Args:
            n_fft (int): STFT的FFT大小，默认为1024。
            hop_length (int): STFT的跳跃长度，默认为256。
            n_channels (int): 初始卷积层的通道数，默认为32。
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # 计算频域中的频率点数
        n = n_fft // 2 + 1
        for _ in range(6):
            # 每次下采样后计算新的频率点数
            n = (n - 1) // 2 + 1

        # 定义一系列卷积层，使用nn.Sequential顺序连接
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),# 输入通道1，输出通道n_channels，卷积核大小7
            nn.LeakyReLU(0.3, inplace=True),  # 应用LeakyReLU激活函数
            ResNet2d(n_channels, 2, stride=(2, 1)),  # ResNet2d模块，通道数乘2，步幅(2,1)
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),  # ResNet2d模块，通道数乘4，步幅(2,2)
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),  # ResNet2d模块，通道数乘4，步幅(2,1)
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),  # ResNet2d模块，通道数乘8，步幅(2,2)
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),  # ResNet2d模块，通道数乘8，步幅(2,1)
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),  # ResNet2d模块，通道数乘16，步幅(2,2)
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))  # 输出通道1，卷积核大小(n,1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法

        Args:
            input (torch.Tensor): 输入波形张量，形状为 [batch_size, 1, sequence_length]

        Returns:
            torch.Tensor: 判别器的输出张量
        """
        # 确保输入通道数为1
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        # 去除通道维度并转换为float32类型，因为torch.stft不接受float16
        x = torch.squeeze(input, 1).to(torch.float32)  # torch.stft() doesn't accept float16
        # 计算STFT
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, onesided=True, return_complex=True)
        # 取复数绝对值
        x = torch.abs(x)
        # 增加通道维度
        x = torch.unsqueeze(x, dim=1)
        # 应用卷积层
        x = self.layers(x)
        return x


class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    """ 
    重建损失函数，但使用STFT而不是梅尔频谱图
    """
    def __init__(self, eps=1e-5):
        """
        初始化 ReconstructionLoss

        Args:
            eps (float): 用于数值稳定的微小常数，默认为1e-5。
        """
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        """
        计算重建损失

        Args:
            input (torch.Tensor): 生成波形，形状为 [batch_size, channels, sequence_length]
            target (torch.Tensor): 目标波形，形状为 [batch_size, channels, sequence_length]

        Returns:
            torch.Tensor: 重建损失值
        """
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            # 计算不同的STFT窗口大小
            s = 2 ** i
            # 计算权重因子
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # 使用STFT而不是64-bin的梅尔频谱图，因为n_fft=64太小，无法支持64个频带
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            # 取复数绝对值
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            # 取复数绝对值
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                # 如果x的长度大于y，则截断x
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                # 如果x的长度小于y，则截断y
                y = y[:, :, :x.shape[-1]]
            # 计算L1损失
            loss += torch.mean(torch.abs(x - y))
            # 计算L2损失
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        # 返回平均损失
        return loss / (12 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    """
    """
    重建损失函数，使用多个梅尔频谱图来计算重建损失。
    """
    def __init__(self, sample_rate, eps=1e-5):
        """
        初始化 ReconstructionLoss2

        Args:
            sample_rate (int): 音频采样率，用于计算梅尔频谱图
            eps (float): 用于数值稳定的微小常数，默认为1e-5
        """
        super().__init__()
        import torchaudio
        self.layers = nn.ModuleList()  # 用于存储梅尔频谱图转换层
        self.alpha = []  # 用于存储每个频谱图的权重因子
        self.eps = eps  # 数值稳定的微小常数

        # 遍历不同的STFT窗口大小，从2^6到2^11
        for i in range(6, 12):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(2 ** i),
                win_length=int(2 ** i),
                hop_length=int(2 ** i / 4),
                n_mels=64)
            # 添加梅尔频谱图转换层
            self.layers.append(melspec)
            # 计算并添加权重因子
            self.alpha.append((2 ** i / 2) ** 0.5)

    def forward(self, input, target):
        """
        计算重建损失

        Args:
            input (torch.Tensor): 生成波形，形状为 [batch_size, channels, sequence_length]
            target (torch.Tensor): 目标波形，形状为 [batch_size, channels, sequence_length]

        Returns:
            torch.Tensor: 重建损失值
        """
        # 初始化总损失
        loss = 0

        # 遍历每个梅尔频谱图转换层及其对应的权重因子
        for alpha, melspec in zip(self.alpha, self.layers):
            # 计算生成波形的梅尔频谱图
            x = melspec(input)
            # 计算目标波形的梅尔频谱图
            y = melspec(target)
            # 如果生成波形的梅尔频谱图长度大于目标波形
            if x.shape[-1] > y.shape[-1]:
                # 截断生成波形的梅尔频谱图
                x = x[:, y.shape[-1]]
            # 如果生成波形的梅尔频谱图长度小于目标波形
            elif x.shape[-1] < y.shape[-1]:
                # 截断目标波形的梅尔频谱图
                y = y[:, x.shape[-1]]
            # 计算L1损失
            loss += torch.mean(torch.abs(x - y))
            # 计算L2损失
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        # 返回总损失
        return loss


class StreamableModel(pl.LightningModule):
    """
    可流式处理的模型，继承自 PyTorch Lightning 的 LightningModule
    """
    def __init__(
        self,
        n_channels: int = 32,
        num_quantizers: int = 8,
        num_embeddings: int = 1024,
        padding: str = "valid",
        batch_size: int = 32,
        sample_rate: int = 24_000,
        segment_length: int = 32270,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech'
    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        """
        初始化 StreamableModel

        Args:
            n_channels (int): 编码器和解码器的通道数，默认为32
            num_quantizers (int): 量化器的数量，默认为8
            num_embeddings (int): 嵌入向量的数量，默认为1024
            padding (str): 卷积层的填充方式，默认为"valid"
            batch_size (int): 批量大小，默认为32
            sample_rate (int): 采样率，默认为24000
            segment_length (int): 音频片段长度，默认为32270
            lr (float): 学习率，默认为1e-4
            b1 (float): Adam优化器的beta1参数，默认为0.5
            b2 (float): Adam优化器的beta2参数，默认为0.9
            dataset (str): 数据集名称，默认为'librispeech'
        """
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        self.automatic_optimization = False  # 关闭自动优化器
        self.encoder = Encoder(n_channels, padding)  # 初始化编码器
        self.decoder = Decoder(n_channels, padding)  # 初始化解码器
        self.quantizer = ResidualVectorQuantizer(  # 初始化量化器
            num_quantizers, num_embeddings, n_channels * 16)

        # 初始化波形判别器列表
        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(resolution=1),
            WaveDiscriminator(resolution=2),
            WaveDiscriminator(resolution=4)
        ])

        # 初始化重建损失函数
        self.rec_loss = ReconstructionLoss()
        # 初始化STFT判别器
        self.stft_discriminator = STFTDiscriminator()

    def configure_optimizers(self):
        """
        配置优化器

        Returns:
            List: 包含生成器和判别器的优化器列表
        """
        lr = self.hparams.lr  # 获取学习率
        b1 = self.hparams.b1  # 获取beta1参数
        b2 = self.hparams.b2  # 获取beta2参数

        # 初始化生成器的Adam优化器
        optimizer_g = torch.optim.Adam(
            # 使用chain将编码器和解码器的参数连接起来
            chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ),
            lr=lr, betas=(b1, b2))
        
        # 初始化判别器的Adam优化器
        optimizer_d = torch.optim.Adam(
            # 使用chain将波形判别器和STFT判别器的参数连接起来
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))
        # 返回优化器列表
        return [optimizer_g, optimizer_d], []

    def forward(self, input):
        """
        前向传播方法

        Args:
            input (torch.Tensor): 输入波形，形状为 [batch_size, channels, sequence_length]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 生成波形，量化码，量化损失
        """
        x = self.encoder(input)  # 输入波形通过编码器
        x = torch.transpose(x, -1, -2)  # 转置张量维度
        x, codes, codebook_loss = self.quantizer(x)  # 通过量化器进行量化
        x = torch.transpose(x, -1, -2)  # 转置回原始维度
        x = self.decoder(x)  # 通过解码器生成波形
        # 返回生成波形，量化码和量化损失
        return x, codes, codebook_loss

    def training_step(self, batch, batch_idx):
        """
        训练步骤

        Args:
            batch (torch.Tensor): 输入批次数据
            batch_idx (int): 当前批次的索引

        Returns:
            torch.Tensor: 损失值
        """
        # 获取生成器和判别器的优化器
        optimizer_g, optimizer_d = self.optimizers()
        input = batch[:, None, :] # 输入数据，形状为 [batch, channel, sequence]

        # train generator
        # 训练生成器

        # 切换到生成器的优化器
        self.toggle_optimizer(optimizer_g)
        # 前向传播获取生成波形和量化损失
        output, _, q_loss = self.forward(input)
        # output: [batch, channel, sequence]
        # print(input.shape, output.shape)

        # 对生成波形进行STFT判别
        stft_out = self.stft_discriminator(output)
        # 计算生成器STFT损失
        g_stft_loss = torch.mean(torch.relu(1 - stft_out))
        # 记录生成器STFT损失
        self.log("g_stft_loss", g_stft_loss)

        # 初始化生成器波形损失
        g_wave_loss = 0
        # 初始化生成器特征损失
        g_feat_loss = 0

        # 遍历三个波形判别器
        for i in range(3):
            # 对输入波形进行判别
            feats1 = self.wave_discriminators[i](input)
            # 对生成波形进行判别
            feats2 = self.wave_discriminators[i](output)
            # 确保特征图数量相同
            assert len(feats1) == len(feats2)
            # 计算波形损失
            g_wave_loss += torch.mean(torch.relu(1 - feats2[-1]))
            g_feat_loss += sum(torch.mean(
                # 计算特征损失
                torch.abs(f1 - f2))
                for f1, f2 in zip(feats1[:-1], feats2[:-1])) / (len(feats1) - 1)
        # 记录平均波形损失
        self.log("g_wave_loss", g_wave_loss / 3)
        # 记录平均特征损失
        self.log("g_feat_loss", g_feat_loss / 3)

        # 计算重建损失
        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        # 记录重建损失
        self.log("g_rec_loss", g_rec_loss, prog_bar=True)

        # 平均特征损失
        g_feat_loss = g_feat_loss / 3
        # 平均对抗损失
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        # 计算总生成器损失
        g_loss = g_adv_loss + 100 * g_feat_loss + g_rec_loss
        # 记录量化损失
        self.log("q_loss", q_loss, prog_bar=True)
        # 记录生成器总损失
        self.log("g_loss", g_loss, prog_bar=True)

        # 反向传播生成器损失
        self.manual_backward(g_loss + q_loss)
        # 更新生成器参数
        optimizer_g.step()
        optimizer_g.zero_grad()
        # 切换回之前的优化器
        self.untoggle_optimizer(optimizer_g)

        # 计算量化码的熵
        codes_entropy = self.quantizer.calc_entropy()
        # 记录量化码熵
        self.log("codes_entropy", codes_entropy, prog_bar=True)

        # train discriminator
        # 训练判别器

        # 切换到判别器的优化器
        self.toggle_optimizer(optimizer_d)
        # 前向传播获取生成波形
        output, _, _ = self.forward(input)

        # 对输入波形进行STFT判别
        stft_out = self.stft_discriminator(input)
        # 计算判别器STFT损失
        d_stft_loss = torch.mean(torch.relu(1 - stft_out))
        # 对生成波形进行STFT判别
        stft_out = self.stft_discriminator(output)
        # 计算判别器STFT损失
        d_stft_loss += torch.mean(torch.relu(1 + stft_out))

        # 初始化判别器波形损失
        d_wave_loss = 0
        # 遍历三个波形判别器
        for i in range(3):
            # 对输入波形进行判别
            feats = self.wave_discriminators[i](input)
            # 计算波形损失
            d_wave_loss += torch.mean(torch.relu(1 - feats[-1]))
            # 对生成波形进行判别
            feats = self.wave_discriminators[i](output)
            # 计算波形损失
            d_wave_loss += torch.mean(torch.relu(1 + feats[-1]))

        # 计算总判别器损失
        d_loss = (d_stft_loss + d_wave_loss) / 4

        # 记录判别器STFT损失
        self.log("d_stft_loss", d_stft_loss)
        # 记录判别器波形损失
        self.log("d_wave_loss", d_wave_loss / 3)

        # 计算总判别器损失
        d_loss = (d_stft_loss + d_wave_loss) / 4
        # 记录判别器总损失
        self.log("d_loss", d_loss, prog_bar=True)

        # 反向传播判别器损失
        self.manual_backward(d_loss)
        # 更新判别器参数
        optimizer_d.step()
        optimizer_d.zero_grad()
        # 切换回之前的优化器
        self.untoggle_optimizer(optimizer_d)

        # 替换向量
        num_replaced = self.quantizer.replace_vectors()
        # 记录替换向量数量
        self.log("num_replaced", float(num_replaced), prog_bar=True)

    def train_dataloader(self):
        # 获取训练数据加载器
        return self._make_dataloader(True)

    def _make_dataloader(self, train: bool):
        import torchaudio

        def collate(examples):
            # 堆叠示例数据
            return torch.stack(examples)

        class VoiceDataset(torch.utils.data.Dataset):
            """
            自定义语音数据集类，用于加载和预处理音频数据。
            """
            def __init__(self, dataset, sample_rate, segment_length):
                """
                初始化 VoiceDataset

                Args:
                    dataset (torch.utils.data.Dataset): 原始音频数据集
                    sample_rate (int): 目标采样率，用于重新采样音频
                    segment_length (int): 每个音频片段的目标长度（样本数）
                """
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                """
                获取指定索引的音频数据并进行预处理

                Args:
                    index (int): 数据索引

                Returns:
                    torch.Tensor: 预处理后的音频数据
                """
                import random
                # 从原始数据集中获取音频数据及其采样率
                x, sample_rate, *_ = self._dataset[index]
                # 将音频重新采样到目标采样率
                x = torchaudio.functional.resample(x, sample_rate, self._sample_rate)

                # 确保音频为单声道
                assert x.shape[0] == 1
                # 去除单声道维度
                x = torch.squeeze(x)
                # 归一化音频幅度，防止溢出
                x *= 0.95 / torch.max(x)

                # 确保音频是一维的
                assert x.dim() == 1
                if x.shape[0] < self._segment_length:
                    # 如果音频长度小于目标片段长度，则在末尾填充零
                    x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
                # 随机选择一个起始位置
                pos = random.randint(0, x.shape[0] - self._segment_length)
                # 截取指定长度的音频片段
                x = x[pos:pos + self._segment_length]
                # 返回预处理后的音频片段
                return x

            def __len__(self):
                """
                获取数据集的大小

                Returns:
                    int: 数据集的大小
                """
                return len(self._dataset)

        if self.hparams.dataset == 'yesno':
            # 如果数据集类型为 'yesno'，则加载 YESNO 数据集
            ds = torchaudio.datasets.YESNO("./data", download=True)
        elif self.hparams.dataset == 'librispeech-dev':
            # 如果数据集类型为 'librispeech-dev'，则加载 Librispeech 开发集
            ds = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean")
        elif self.hparams.dataset == 'librispeech':
            # 如果数据集类型为 'librispeech'，则根据训练模式加载训练集或开发集
            url = "train-clean-100" if train else "dev-clean"
            ds = torchaudio.datasets.LIBRISPEECH("./data", url=url)
        else:
            # 如果数据集类型不支持，则抛出值错误
            raise ValueError()
        
        # 将原始数据集包装为 VoiceDataset，并设置目标采样率和片段长度
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        # 创建数据加载器，设置批量大小、是否打乱数据以及自定义的 collate 函数
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate)
        # 返回数据加载器
        return loader


class KMeanCodebookInitCallback(pl.Callback):
    """
    回调类，用于在模型训练开始时对量化器的代码本进行 K-Means 初始化。
    """
    def on_fit_start(self, trainer, model):
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer
        # run the k-means
        # algorithm on the first training batch and use the learned
        # centroids as initialization
        """
        当训练开始时调用的方法。

        Args:
            trainer (pl.Trainer): PyTorch Lightning 的 Trainer 对象。
            model (StreamableModel): 正在训练的模型实例。
        """
        # 在第一个训练批次上运行 K-Means 算法，并使用学习到的质心作为初始化
        # 获取第一个训练批次
        batch = next(iter(model.train_dataloader()))
        # 将批次数据移动到模型的设备上，并增加一个维度
        input = batch[:, None, :].to(model.device)
        with torch.no_grad():
            x = torch.flatten(model.encoder(input))  # 将编码器的输出展平
            mean = torch.mean(x, axis=0)  # 计算均值
            std = torch.std(x, axis=0)  # 计算标准差
            # 使用计算得到的均值和标准差初始化量化器的权重
            torch.nn.init.normal_(model.quantizer.weight, mean=mean, std=std)  
        print(f"KMeanCodebookInitCallback {mean} {std}")


def train():
    """
    训练函数，用于初始化模型和训练器并开始训练。

    Returns:
        StreamableModel: 训练好的模型实例。
    """
    # 初始化模型，假设使用 LibriTTS 数据集，批量大小为32，采样率为16000，片段长度为32270，填充方式为'same'
    model = StreamableModel(
        batch_size=32,
        sample_rate=16_000,
        segment_length=32270,
        padding='same',
        dataset='librispeech')
    
    # 初始化训练器，设置最大训练周期为10000，使用16位混合精度训练，记录日志到当前目录下的CSV文件
    trainer = pl.Trainer(
        max_epochs=10000,
        log_every_n_steps=2,
        precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
        callbacks=[
            # 模型检查点回调，每50000步保存一次
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=50000),
            # K-Means 初始化回调
            KMeanCodebookInitCallback(),
        ],
    )

    # 开始训练
    trainer.fit(
        model,
    )

    return model



if __name__ == "__main__":
    
    train()

