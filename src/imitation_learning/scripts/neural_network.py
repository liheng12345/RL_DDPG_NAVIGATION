import torch
import torch.nn as nn


class PoseToControl(nn.Module):
    def __init__(self, control_dim):
        super(PoseToControl, self).__init__()

        self.in_to_lstm = nn.Sequential(
            nn.Linear(19, 256),
            nn.ReLU(),  # 这里的19是雷达数据360/30+odom数据7=19
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, control_dim))

    def forward(self, odom_input, laser_scan):
        total_input = torch.cat((odom_input, laser_scan), dim=1)
        # print(total_input)
        h = self.in_to_lstm(total_input)
        return h