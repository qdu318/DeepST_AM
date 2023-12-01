import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TCN import TemporalConvNet


def cal_linear_num(layer_num, num_timesteps_input):
    result = num_timesteps_input + 4 * (2**layer_num - 1)
    return result


class TCN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            channel_size,
            layer_num,
            num_timesteps_input,
            kernel_size=3
    ):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=in_channels, num_channels=channel_size, kernel_size=kernel_size)
        linear_num = cal_linear_num(layer_num, num_timesteps_input)
        self.linear = nn.Linear(linear_num, out_channels)
        self.linear2 = nn.Linear(4, 3)

    def forward(self, X):
        X = X.permute(0, 2, 1, 3)
        X = self.tcn(X)
        X = self.linear(X)
        X = X.permute(0, 2, 1, 3)
        return X


def cal_channel_size(layers, timesteps_input):
    channel_size = []
    for i in range(layers - 1):
        channel_size.append(timesteps_input)
    channel_size.append(timesteps_input - 2)
    return channel_size


class DeepTRI(nn.Module):
    def __init__(
            self,
            num_nodes,
             out_channels,
             spatial_channels,
             features,
             timesteps_input,
             timesteps_output
    ):
        super(DeepTRI, self).__init__()
        self.spatial_channels = spatial_channels
        tcn_layer = 5
        channel_size = cal_channel_size(tcn_layer, timesteps_input)
        self.tcn1 = TCN(
                        in_channels=features,
                        out_channels=out_channels,
                        channel_size=channel_size,
                        layer_num=tcn_layer,
                        num_timesteps_input=timesteps_input
                    )
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        channel_size = cal_channel_size(tcn_layer, timesteps_input - 2)
        self.tcn2 = TCN(
                        in_channels=spatial_channels,
                        out_channels=out_channels,
                        channel_size=channel_size,
                        layer_num=tcn_layer,
                        num_timesteps_input=timesteps_input - 2
                    )

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()
        self.fully = nn.Linear(32, timesteps_output)
        self.matchconv = nn.Conv2d(in_channels=4, out_channels=spatial_channels*2, kernel_size=(3, 1), stride=1, bias=True)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, W_nodes, X):
        t = self.tcn1(X)
        lfs = torch.einsum("ij,jklm->kilm", [W_nodes, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))  # [5,29,3,16]

        # V_asist = V_asist.permute((0, 3, 2, 1))
        # V_asist_match = self.matchconv(V_asist).permute((0, 3, 2, 1))
        gate_spatail = F.relu(t2)
        # gate_spatail = F.relu(torch.add(V_asist_match[:, :, :, 0:self.spatial_channels], t2))
        # Glu_out = torch.cat([gate_spatail, V_asist_match[:, :, :, -self.spatial_channels:]], dim=3)
        Glu_out = gate_spatail.permute(0, 1, 3, 2)
        t3 = self.tcn2(Glu_out)

        out3 = self.batch_norm(t3)

        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        return out4


