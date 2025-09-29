import copy

import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=60):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.embed = nn.Embedding(num_classes, nz)
        self.z_chunk_size = nz // 2
        self.proj_z = nn.Linear(self.z_chunk_size, ngf * 2 * self.init_size ** 2)

        self.g_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])

        self.attn_block = SelfAttention(ngf * 2)

        self.proj_o = nn.Sequential(
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.weights_init()

    def forward(self, z, labels):
        label_embedding = self.embed(labels.long())
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1)

        zs = torch.split(z, self.z_chunk_size, dim=1)
        z = zs[0]
        # print(z.shape)
        ys = [torch.cat([label_embedding, z_chunk.unsqueeze(-1).unsqueeze(-1)], dim=1) for z_chunk in zs[1:]]

        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.init_size, self.init_size)

        for idx, g_block in enumerate(self.g_blocks):
            h = g_block(h)
            # print(h.shape)
            if idx < len(ys):
                conv_layer = nn.Conv2d(384, 128, kernel_size=1).cuda()
                h = h + conv_layer(ys[idx])
            if idx == 0:
                h = self.attn_block(h)

        img = self.proj_o(h)
        return img

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def copy(self):
        clone = copy.deepcopy(self)
        return clone.cuda()

    def print_param(self):
        params = 0
        for module in self.modules():
            params += sum([p.data.nelement() for p in module.parameters()])

        print(
            "Generator's trainable parameters:  {:.0f}M {:.0f}K {:d}".format(
                params // 1e6, (params // 1e3) % 1000, params % 1000
            )
        )

        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print("Model Generator's size: {:.3f}MB".format(size_all_mb))


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        key = self.key(x).view(batch_size, -1, width * height)  # B x C x N
        attention = torch.bmm(query, key)  # B x N x N
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(batch_size, -1, width * height)  # B x C x N
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# class AttentionBlock(nn.Module):
#     """
#     AttentionBlock Class
#     Values:
#     channels: number of channels in input
#     """
#
#     def __init__(self, channels):
#         super().__init__()
#
#         self.channels = channels
#
#         self.theta = nn.utils.spectral_norm(
#             nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False)
#         )
#         self.phi = nn.utils.spectral_norm(
#             nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False)
#         )
#         self.g = nn.utils.spectral_norm(
#             nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=False)
#         )
#         self.o = nn.utils.spectral_norm(
#             nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False)
#         )
#
#         self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)
#
#     def forward(self, x):
#         spatial_size = x.shape[2] * x.shape[3]
#
#         # Apply convolutions to get query (theta), key (phi), and value (g) transforms
#         theta = self.theta(x)
#         phi = F.max_pool2d(self.phi(x), kernel_size=2)
#         g = F.max_pool2d(self.g(x), kernel_size=2)
#
#         # Reshape spatial size for self-attention
#         theta = theta.view(-1, self.channels // 8, spatial_size)
#         phi = phi.view(-1, self.channels // 8, spatial_size // 4)
#         g = g.view(-1, self.channels // 2, spatial_size // 4)
#
#         # Compute dot product attention with query (theta) and key (phi) matrices
#         beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)
#
#         # Compute scaled dot product attention with value (g) and attention (beta) matrices
#         o = self.o(
#             torch.bmm(g, beta.transpose(1, 2)).view(
#                 -1, self.channels // 2, x.shape[2], x.shape[3]
#             )
#         )
#
#         # Apply gain and residual
#         return self.gamma * o + x
