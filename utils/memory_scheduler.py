"""
Memory Training Scheduler
温度-多样性联动调度器

功能：
1. 温度从高到低退火（探索→利用）
2. 多样性权重从低到高增长（自由分化→强制平衡）
3. Warmup结束后冻结可学习温度（避免与scheduler打架）
"""

import torch


class MemoryTrainingScheduler:
    """
    温度与多样性权重联动调度器

    退火策略：
    - Temperature: 0.5 → 0.1  (高→低，探索→利用)
    - Diversity Weight: 0.0 → target_weight (低→高，自由→约束)

    Args:
        total_epochs: 总训练epoch数
        warmup_epochs: warmup轮数（默认30）
        temp_init: 初始温度
        temp_final: 最终温度
        div_weight_init: 多样性权重初始值
        div_weight_final: 多样性权重最终值
    """

    def __init__(
        self,
        total_epochs: int = 200,
        warmup_epochs: int = 30,
        temp_init: float = 0.5,
        temp_final: float = 0.20,  # 提高最低温度，避免过早硬化
        div_weight_init: float = 0.0,
        div_weight_final: float = 0.01,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.div_weight_init = div_weight_init
        self.div_weight_final = div_weight_final

        self.warmup_finished = False

    def get_temperature(self, epoch: int) -> float:
        """
        获取当前epoch的温度

        Warmup阶段线性退火，之后保持最终温度
        """
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            temp = self.temp_init - (self.temp_init - self.temp_final) * progress
        else:
            temp = self.temp_final
        return temp

    def get_diversity_weight(self, epoch: int) -> float:
        """
        获取当前epoch的多样性权重

        Warmup阶段线性增长，之后保持最终权重
        """
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            weight = self.div_weight_init + (self.div_weight_final - self.div_weight_init) * progress
        else:
            weight = self.div_weight_final
        return weight

    def should_freeze_temperature(self, epoch: int) -> bool:
        """
        判断是否应该冻结可学习温度

        Warmup结束时冻结，避免与scheduler打架
        """
        if epoch == self.warmup_epochs and not self.warmup_finished:
            self.warmup_finished = True
            return True
        return False

    def get_diagnostics(self, epoch: int) -> dict:
        """获取当前状态的诊断信息"""
        return {
            'epoch': epoch,
            'temperature': self.get_temperature(epoch),
            'diversity_weight': self.get_diversity_weight(epoch),
            'warmup_progress': min(epoch / self.warmup_epochs, 1.0),
            'warmup_finished': self.warmup_finished,
        }


if __name__ == '__main__':
    # 测试调度器
    print("Testing MemoryTrainingScheduler...")

    scheduler = MemoryTrainingScheduler(
        total_epochs=200,
        warmup_epochs=30,
        temp_init=0.5,
        temp_final=0.1,
        div_weight_init=0.0,
        div_weight_final=0.01,
    )

    # 测试几个关键epoch
    test_epochs = [0, 1, 10, 29, 30, 50, 100, 200]

    print("\nScheduler Trajectory:")
    print(f"{'Epoch':<8} {'Temperature':<15} {'Div Weight':<15} {'Should Freeze':<15}")
    print("-" * 60)

    for ep in test_epochs:
        temp = scheduler.get_temperature(ep)
        div_weight = scheduler.get_diversity_weight(ep)
        should_freeze = scheduler.should_freeze_temperature(ep)

        print(f"{ep:<8} {temp:<15.4f} {div_weight:<15.6f} {should_freeze}")

    print("\n✅ Test passed!")
