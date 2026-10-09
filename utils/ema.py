"""Exponential moving average of the model weights.

扩散模型常用：评测时用这份滑动平均权重，结果比训练当下的权重稳。衰减调度照抄原版
diffusion_policy —— 一开始 decay≈0（等于直接拷贝），之后逐渐逼近 max_value。
"""

import copy

import torch


class EMA:
    def __init__(self, model, power=0.75, max_value=0.9999):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.power = power
        self.max_value = max_value
        self.step_count = 0

    def step(self, model):
        decay = min(1 - (1 + self.step_count) ** -self.power, self.max_value)
        self.step_count += 1
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                # decay × 旧影子权重 + (1 − decay) × 当前训练权重
                # 原版压成一行是 ema_p.mul_(decay).add_(p, alpha=1 - decay)
                ema_p.mul_(decay)                 # 影子权重 × decay
                ema_p.add_(p, alpha=1 - decay)    # 加上 (1 - decay) × 当前权重

    def state_dict(self):
        return self.model.state_dict()
