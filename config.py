from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    dataset_root: str = r"C:\Users\srush\OneDrive\Documents\Honors\Datasets\LEVIR-CD256"
    img_size: Tuple[int, int] = (256, 256)

    batch_size: int = 8
    num_workers: int = 2

    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Deformable SSA
    ssa_heads: int = 8
    ssa_reduce_ratio: int = 2
    ssa_max_offset: float = 2.0

    # Offset regularization
    lambda_offset: float = 0.01

    # LR warmup
    warmup_epochs: int = 5

    # Misc
    seed: int = 1337
    amp: bool = True                                                                                                                                                                                                                
