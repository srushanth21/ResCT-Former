from pathlib import Path
from .base_cd_dataset import BaseChangeDataset

class LEVIRDataset(BaseChangeDataset):
    def __init__(self, root, split="train", transform=None):
        root = Path(root) / split

        pre_dir = root / "A"
        post_dir = root / "B"
        label_dir = root / "label"

        super().__init__(
            pre_dir=pre_dir,
            post_dir=post_dir,
            label_dir=label_dir,
            transform=transform
        )
