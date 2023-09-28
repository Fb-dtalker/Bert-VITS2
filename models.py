import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from commons import init_weights, get_padding
from text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = self.drop(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

# 音素的时长预测器
class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        # x:文本的隐条件，也就是TextEncoder编码后的结果
        # 输出的是
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1) #弄出u, v的部分，其中v的部分因为是我们引入的舍弃掉
            logw = z0 #这个就是预测出来的时长的离散分布，向上取整得到预测的时长
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels) # 构建一个(全部语言的音素, hidden_channels)shape的词嵌入模型 (batchsize, 序列长度)->(batchsize, 序列长度, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5) # 正则初始化词嵌入模型的权重
        self.tone_emb = nn.Embedding(num_tones, hidden_channels) # 构建一个(全部声调, hidden_channel)shape的词嵌入模型
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5) # 正则初始化词嵌入模型的权重
        self.language_emb = nn.Embedding(num_languages, hidden_channels) # 构建一个(全部语种, hidden_channels)shape的词嵌入模型
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5) # 正则初始化
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1) # (1024, 序列长度) 卷积hidden_channels个 (1024, 1) -> (hidden_channels, 序列长度)
        self.ja_bert_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 音素序列, 音素序列长度(shape是：[batchsize, 长度数值]), 声调序列, 语言序列, 中文bert结果, 日文bert结果, embedding词典表
    def forward(self, x, x_lengths, tone, language, bert, ja_bert, g=None):
        bert_emb = self.bert_proj(bert).transpose(1, 2) #(batchsize, 1024, 序列长度) -> (batchsize, hidden_channels, 序列长度) -> (batchsize, 序列长度, hidden_channels)
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2) #(batchsize, 768, 序列长度) -> (batchsize, hidden_channels, 序列长度) -> (batchsize, 序列长度, hidden_channels)
        # x这里相当于融合了音素嵌入、声调嵌入、语言嵌入、bert中文预测、bert日文预测的结果
        x = (
            self.emb(x) # (batchsize, 序列长度, hidden_channels)
            + self.tone_emb(tone) # (batchsize, 序列长度, hidden_channels)
            + self.language_emb(language) # (batchsize, 序列长度, hidden_channels)
            + bert_emb # (batchsize, 序列长度, hidden_channels)
            + ja_bert_emb # (batchsize, 序列长度, hidden_channels)
        ) * math.sqrt(
            #embedding的常规操作，embedding是分布是N(0,1)，但是embedding matrix 初始化后的方差为1/embedding_size，
            #所以乘上sqrt(embedding_size)能够时其分布回到N(0,1)，加速训练时的收敛
            #https://zhuanlan.zhihu.com/p/442509602
            self.hidden_channels 
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t] (batchsize, hidden_channels, 序列长度)
        # 对x中x_lenghts超出max_lenght设置的长度的部分进行遮蔽，也就是说x_lenghts在设计上是可以小于实际x的长度的
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        # https://www.bilibili.com/video/BV1x94y1r7WG/?p=61
        # 多头注意力层
        x = self.encoder(x * x_mask, x_mask, g=g) #输入(batchsize,192,序列长度)，返回(batchsize,192,序列长度)
        # 将多头注意力层的结果进行mask后用1维卷积核拉通
        stats = self.proj(x) * x_mask

        # 结果分为均值部分和log(方差)部分
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # h_text, 均值, 方差, mask
        return x, m, logs, x_mask

# Flow模块，将音频采样出来的分布转换为文本采样出来的分布，或将文本采样出来的分布转为音频采样出来的分布
# 支持前后两个方向的数据流动，一般而言：
# 训练时 Flow(音频采样分布) -> 文本采样分布
# 推理时 Flow(文本采样分布) -> 音频采样分布
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

#后验编码器，把音频编码成一个标准分布
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        #输入层，主要是为了调整层数和大小
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        #WavNet，对音频做特征提取
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        #输出层，主要也是为了调整输出大小
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        #先把梅罗普数据进行拉通层一个(hidden_channels, 序列长度)的矩阵，然后进行mask
        x = self.pre(x) * x_mask
        #用WavNet进行音频特征提取
        x = self.enc(x, x_mask, g=g)
        #特征提取结果再次拉通，并且进行mask
        stats = self.proj(x) * x_mask
        #将结果分层均值和Log(标准差)两部分
        m, logs = torch.split(stats, self.out_channels, dim=1)
        #对结果进行采样，得到 均值+(均值标准差分布的随机数*标准差) ,再进行mask
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        #采样值, 均值, 标准差, mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=4,
        flow_share_parameter=False,
        use_transformer_flow=True,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid, tone, language, bert, ja_bert):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)

        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging

        l_length = l_length_dp + l_length_sdp

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
        )

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
    ):
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
        # g = self.gst(y)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        #exp(logw)就是得到了最终每个音素的长度，length_scale可以对音素进行整体的缩放
        #那这样我们对其中一部分进行缩放就能够实现拖长声音的效果
        w = torch.exp(logw) * x_mask * length_scale
        print(w)
        w[0][0][3:4] *= 4
        w[0][0][5:6] *= 4
        print(w)
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
