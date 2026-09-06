import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm


class CachedSamplerBase(Sampler):
    def __init__(self, dataset, patch_size, cache_prefix):
        cache_dir = Path("dataset/cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        # cache file paths
        self.pos_file = cache_dir / f"{cache_prefix}_pos_{patch_size}.npy"
        self.neg_file = cache_dir / f"{cache_prefix}_neg_{patch_size}.npy"
        self.map_file = cache_dir / f"{cache_prefix}_event_map_{patch_size}.pkl"

        self.dataset = dataset

        # load or build positive/negative sample indices
        if self.pos_file.exists() and self.neg_file.exists():
            self.positive_indices = np.load(self.pos_file).tolist()
            self.negative_indices = np.load(self.neg_file).tolist()
        else:
            self.positive_indices = []
            self.negative_indices = []

            was_transform = dataset.apply_transform
            try:
                dataset.apply_transform = False  # disable transforms for scanning
                for i in tqdm(range(len(dataset)), desc="Scanning dataset for positives"):
                    sample = dataset[i]
                    if sample["mask"].any():
                        self.positive_indices.append(i)
                    else:
                        self.negative_indices.append(i)
            finally:
                dataset.apply_transform = was_transform

            print(f"[CachedSamplerBase] Found {len(self.positive_indices)} positive and {len(self.negative_indices)} negative samples.")

            np.save(self.pos_file, np.asarray(self.positive_indices, dtype=np.int32))
            np.save(self.neg_file, np.asarray(self.negative_indices, dtype=np.int32))

        if not self.positive_indices:
            raise ValueError("No positive samples found in dataset!")


class BalancedPosNegSampler(CachedSamplerBase):
    def __init__(self, dataset, patch_size):
        super().__init__(dataset, patch_size, cache_prefix="train")

        # load or build event → negative_indices map (with string keys)
        if self.map_file.exists():
            with open(self.map_file, "rb") as f:
                self.event_to_neg_indices = pickle.load(f)
        else:
            tmp = defaultdict(list)
            for idx in tqdm(self.negative_indices, desc="Grouping negatives by event"):
                event = dataset[idx]["event"]  # use string name
                tmp[event].append(idx)

            self.event_to_neg_indices = {
                e: np.asarray(v, dtype=np.int32) for e, v in tmp.items()
            }

            with open(self.map_file, "wb") as f:
                pickle.dump(self.event_to_neg_indices, f)

        self.events = list(self.event_to_neg_indices.keys())

    def __iter__(self):
        num_pos = len(self.positive_indices)
        num_events = len(self.events)

        # Deterministic order to distribute the remainder fairly
        events = sorted(self.events)

        negs_per_event = num_pos // num_events
        remainder = num_pos % num_events

        sampled_neg_parts = []
        collected = 0

        # First pass: take without replacement, capped by pool size
        for i, event in enumerate(events):
            n = negs_per_event + (1 if i < remainder else 0)
            pool = self.event_to_neg_indices[event]
            take = min(n, len(pool))
            if take > 0:
                picks = np.random.choice(pool, take, replace=False)
                sampled_neg_parts.append(picks)
                collected += take

        # Top up if some events had too few negatives
        if collected < num_pos:
            deficit = num_pos - collected
            all_negs = np.concatenate(
                [self.event_to_neg_indices[e] for e in events if len(self.event_to_neg_indices[e]) > 0]
            )
            # Fall back to replacement to guarantee we hit the target
            extra = np.random.choice(all_negs, deficit, replace=True)
            sampled_neg_parts.append(extra)

        sampled_neg = (
            np.concatenate(sampled_neg_parts).astype(np.int32)
            if sampled_neg_parts else np.empty(0, dtype=np.int32)
        )

        combined = np.concatenate([
            np.asarray(self.positive_indices, dtype=np.int32),
            sampled_neg
        ])
        # Ensure tqdm/DataLoader length matches what we actually yield
        assert len(combined) == 2 * num_pos, f"combined={len(combined)} expected={2 * num_pos}"

        np.random.shuffle(combined)

        return iter(combined.tolist())

    def __len__(self):
        return len(self.positive_indices) * 2


class FixedBalancedSampler(CachedSamplerBase):
    def __init__(self, dataset, patch_size: int):
        super().__init__(dataset, patch_size, cache_prefix="val")

        # fixed validation negatives (balanced)
        self.fixed_file = Path("dataset/cache") / f"val_fixed_neg_{patch_size}.npy"
        if self.fixed_file.exists():
            fixed_neg = np.load(self.fixed_file)
        else:
            if len(self.negative_indices) < len(self.positive_indices):
                raise ValueError("Not enough negatives for balanced validation!")

            fixed_neg = np.random.choice(
                np.asarray(self.negative_indices, dtype=np.int32),
                len(self.positive_indices),
                replace=False
            )
            np.save(self.fixed_file, fixed_neg)

        self.fixed_indices = np.concatenate([
            np.asarray(self.positive_indices, dtype=np.int32),
            fixed_neg
        ]).tolist()

    def __iter__(self):
        idxs = self.fixed_indices.copy()
        np.random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.fixed_indices)