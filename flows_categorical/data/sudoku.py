# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""Sudoku dataset for discrete flow matching.

Loads preprocessed sudoku puzzles from numpy files created by build_sudoku_dataset.py.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict, Union


class SudokuDataset(Dataset):
    """Real Sudoku dataset loaded from preprocessed numpy files.

    The data is expected to be preprocessed using scripts/build_sudoku_dataset.py,
    which downloads from HuggingFace and saves as numpy arrays with metadata.

    Directory structure:
        data_dir/
        ├── train/
        │   ├── dataset.json         # Metadata
        │   ├── all__inputs.npy       # Partial puzzles (with blanks)
        │   ├── all__labels.npy       # Complete solutions
        │   └── all__*.npy            # Other metadata arrays
        └── test/
            └── ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        subset: str = 'all',
        return_inputs: bool = False,
        normalize_vocab: bool = True,
    ):
        """Initialize Sudoku dataset.

        Args:
            data_dir: Path to preprocessed data directory
            split: Dataset split ('train' or 'test')
            subset: Data subset name (default: 'all')
            return_inputs: If True, return dict with inputs and labels. If False, return only labels.
            normalize_vocab: If True, subtract 1 to get 0-9 range. If False, keep 1-10 range from preprocessing.
        """
        self.data_dir = data_dir
        self.split = split
        self.subset = subset
        self.return_inputs = return_inputs
        self.normalize_vocab = normalize_vocab

        # Construct paths
        split_dir = os.path.join(data_dir, split)
        metadata_path = os.path.join(split_dir, "dataset.json")
        inputs_path = os.path.join(split_dir, f"{subset}__inputs.npy")
        labels_path = os.path.join(split_dir, f"{subset}__labels.npy")

        # Check if data exists
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Data directory not found: {split_dir}\n"
                f"Please run the preprocessing script first:\n"
                f"  python scripts/build_sudoku_dataset.py preprocess-data --output-dir {data_dir}"
            )

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load data
        if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Data files not found in {split_dir}\n"
                f"Expected: {subset}__inputs.npy and {subset}__labels.npy"
            )

        inputs = np.load(inputs_path)  # Shape: (n_samples, 81)
        labels = np.load(labels_path)  # Shape: (n_samples, 81)

        # Preprocessing adds 1 to shift from 0-9 to 1-10 range
        # We normalize back to 0-9 for standard ML conventions
        if normalize_vocab:
            inputs = inputs - 1  # Now 0-9 range (0 = blank)
            labels = labels - 1  # Now 0-9 range

        # Convert to torch tensors
        self.inputs = torch.from_numpy(inputs).long()
        self.labels = torch.from_numpy(labels).long()

        assert self.inputs.shape == self.labels.shape, \
            f"Inputs and labels must have same shape, got {self.inputs.shape} and {self.labels.shape}"
        assert self.inputs.shape[1] == 81, \
            f"Expected sequence length 81, got {self.inputs.shape[1]}"

        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.inputs)}")
        print(f"  Shape: {self.inputs.shape}")
        print(f"  Vocab range: {self.labels.min().item()}-{self.labels.max().item()}")
        if self.metadata:
            print(f"  Metadata: vocab_size={self.metadata.get('vocab_size')}, "
                  f"seq_len={self.metadata.get('seq_len')}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single sudoku puzzle.

        Returns:
            If return_inputs=False: Complete solution as 1D tensor of length 81 (for flow matching)
            If return_inputs=True: Dict with 'input' (partial puzzle) and 'label' (solution)
        """
        if self.return_inputs:
            return {
                'input': self.inputs[idx],
                'label': self.labels[idx],
            }
        else:
            # For flow matching training, we only need the complete solutions
            return self.labels[idx]


def create_dataloaders(config: 'Config', batch_size: Optional[int] = None):
    """Create training and validation dataloaders.

    Args:
        config: Configuration object
        batch_size: Optional batch size override

    Returns:
        train_loader, val_loader (or test_loader if no val split available)
    """
    if batch_size is None:
        batch_size = config.training.batch_size

    # Create datasets
    train_dataset = SudokuDataset(
        data_dir=config.data.data_dir,
        split='train',
        subset=config.data.subset,
        return_inputs=config.data.return_inputs,
        normalize_vocab=True,  # Always normalize to 0-9 range
    )

    # Use 'test' split as validation (the dataset has train/test, not train/val)
    val_dataset = SudokuDataset(
        data_dir=config.data.data_dir,
        split='test',
        subset=config.data.subset,
        return_inputs=config.data.return_inputs,
        normalize_vocab=True,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader