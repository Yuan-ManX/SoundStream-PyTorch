import torch
import torch.nn as nn

from soundstream import Encoder, Decoder, ResidualVectorQuantizer


class EncoderDecoder(nn.Module):
    """
    编码器-解码器模型，结合了编码器、解码器和残差向量量化器。

    Args:
        n_channels (int, optional): 通道数。默认为 32。
        num_quantizers (int, optional): 量化器的数量。默认为 8。
        num_embeddings (int, optional): 嵌入字典的大小。默认为 1024。
        padding (str, optional): 填充方式。默认为 "valid"。
    """
    def __init__(
        self,
        n_channels: int = 32,
        num_quantizers: int = 8,
        num_embeddings: int = 1024,
        padding: str = "valid"
    ):
        super().__init__()
        # 初始化编码器
        self.encoder = Encoder(n_channels, padding)
        # 初始化解码器
        self.decoder = Decoder(n_channels, padding)
        # 初始化残差向量量化器
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, num_embeddings, n_channels * 16)

    def forward(self, x):
        """
        前向传播方法，编码输入数据。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, length)。

        Returns:
            torch.Tensor: 编码后的张量，形状为 (batch_size, length, num_quantizers)。
        """
        return self.encode(x)


    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        编码输入数据。

        Args:
            input (torch.Tensor): 输入张量，形状为 (batch_size, length)。

        Returns:
            torch.Tensor: 编码后的张量，形状为 (batch_size, length, num_quantizers)。
        """
        assert input.ndim == 2
        # 增加一个维度，形状变为 (batch_size, 1, length)
        x = torch.unsqueeze(input, 1)
        # 通过编码器，形状变为 (batch_size, n_channels, length')
        x = self.encoder(x)
        # 交换最后两个维度，形状变为 (batch_size, length', n_channels)
        x = torch.transpose(x, -1, -2)
        # 通过量化器，得到量化后的张量
        _, codes, _ = self.quantizer(x)
        # 返回量化后的张量，形状为 (batch_size, length', num_quantizers)
        return codes

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        解码量化后的数据。

        Args:
            input (torch.Tensor): 量化后的输入张量，形状为 (batch_size, length, num_quantizers)。

        Returns:
            torch.Tensor: 解码后的张量，形状为 (batch_size, length)。
        """
        # input: [batch_size, length, num_quantizers]
        # 反量化，形状变为 (batch_size, length', n_channels)
        x = self.quantizer.dequantize(input)
        # 交换最后两个维度，形状变为 (batch_size, n_channels, length')
        x = torch.transpose(x, -1, -2)
        # 通过解码器，形状变为 (batch_size, 1, length)
        x = self.decoder(x)
        # 去除第二个维度，形状变为 (batch_size, length)
        x = torch.squeeze(x, 1)
        # 返回解码后的张量
        return x


def soundstream_16khz(pretrained=False, **kwargs):
    """SoundStream encoder decoder
    
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    """
    SoundStream 编码器-解码器模型函数。

    Args:
        pretrained (bool): 是否加载预训练的权重。如果为 True，则加载预训练的模型权重。
        **kwargs: 其他关键字参数。

    Returns:
        EncoderDecoder: 返回一个 EncoderDecoder 模型的实例。
    """
    # 调用 EncoderDecoder 模型
    model = EncoderDecoder()
    # 如果需要加载预训练的权重
    # 使用 torch.hub 从指定的 URL 加载预训练的 state_dict
    # 这里 URL 为空字符串，需要根据实际情况替换为有效的预训练模型权重 URL
    state_dict = torch.hub.load_state_dict_from_url("", map_location='cpu')

    # 加载预训练的权重到模型中
    # strict=False 表示允许模型和预训练权重之间存在部分不匹配
    model.load_state_dict(state_dict['state_dict'], strict=False)
    # 将模型设置为评估模式
    model.eval()
    
    return model
