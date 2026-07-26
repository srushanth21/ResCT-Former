from torch.utils.data import Dataset
from pathlib import Path

from .utils import read_rgb, read_mask


class BaseChangeDataset(Dataset):

    def __init__(
        self,
        pre_dir,
        post_dir,
        label_dir,
        transform=None,
        list_file=None,
        ext=".png"
    ):

        pre_dir = Path(pre_dir)
        post_dir = Path(post_dir)
        label_dir = Path(label_dir)

        self.transform = transform

        # =================================================
        # CHECK DIRECTORIES
        # =================================================

        missing_dirs = [

            str(p)

            for p in (
                pre_dir,
                post_dir,
                label_dir
            )

            if not p.exists()
        ]

        if missing_dirs:

            raise FileNotFoundError(
                "Dataset directories not found: "
                + ", ".join(missing_dirs)
                + " (check DATA_ROOT)"
            )

        # =================================================
        # LOAD USING SPLIT FILE
        # =================================================

        if list_file is not None:

            list_file = Path(list_file)

            if not list_file.exists():

                raise FileNotFoundError(
                    f"Split file not found: {list_file}"
                )

            sample_ids = [

                line.strip()

                for line in list_file
                .read_text(encoding="utf-8")
                .splitlines()

                if line.strip()
            ]

            if len(sample_ids) == 0:

                raise ValueError(
                    f"No samples found in {list_file}"
                )

            self.pre_files = [
                pre_dir / f"{sid}{ext}"
                for sid in sample_ids
            ]

            self.post_files = [
                post_dir / f"{sid}{ext}"
                for sid in sample_ids
            ]

            self.label_files = [
                label_dir / f"{sid}{ext}"
                for sid in sample_ids
            ]

        # =================================================
        # AUTO SCAN FOLDERS
        # =================================================

        else:

            self.pre_files = sorted(
                pre_dir.glob(f"*{ext}")
            )

            self.post_files = sorted(
                post_dir.glob(f"*{ext}")
            )

            self.label_files = sorted(
                label_dir.glob(f"*{ext}")
            )

        # =================================================
        # EMPTY DATASET CHECK
        # =================================================

        if len(self.pre_files) == 0:

            raise ValueError(
                "Dataset is empty (0 samples). "
                f"Searched pre={pre_dir}"
            )

        # =================================================
        # SIZE CHECK
        # =================================================

        if not (
            len(self.pre_files)
            ==
            len(self.post_files)
            ==
            len(self.label_files)
        ):

            raise ValueError(
                "Dataset size mismatch:\n"
                f"pre={len(self.pre_files)}\n"
                f"post={len(self.post_files)}\n"
                f"label={len(self.label_files)}"
            )

        # =================================================
        # FILE EXISTENCE CHECK
        # =================================================

        for p in (
            self.pre_files[0],
            self.post_files[0],
            self.label_files[0]
        ):

            if not p.exists():

                raise FileNotFoundError(
                    f"Expected file not found: {p}"
                )

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return len(self.pre_files)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, idx):

        pre = read_rgb(self.pre_files[idx])

        post = read_rgb(self.post_files[idx])

        mask = read_mask(self.label_files[idx])

        if self.transform:

            pre, post, mask = self.transform(
                pre,
                post,
                mask
            )

        return pre, post, mask