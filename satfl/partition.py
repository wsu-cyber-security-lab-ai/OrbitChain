from typing import List, Dict, Tuple
import numpy as np

from torch.utils.data import DataLoader, Subset

from .config import DEFAULT_BATCH_SIZE, NUM_WORKERS, SEED
from .datasets import _get_labels_array


def iid_partition_indices(train_indices: np.ndarray, num_clients: int) -> List[np.ndarray]:
    shuffled = train_indices.copy()
    np.random.shuffle(shuffled)
    return list(np.array_split(shuffled, num_clients))


def dirichlet_partition_indices(
    train_indices: np.ndarray,
    labels_all: np.ndarray,
    num_clients: int,
    alpha: float = 0.1,
) -> List[np.ndarray]:
    """
    Dirichlet label-skew non-IID partition over train_indices.
    labels_all: full labels array indexed by original dataset indices.
    """
    y_train = labels_all[train_indices]
    classes = np.unique(y_train)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = train_indices[y_train == c]
        np.random.shuffle(idx_c)
        n = len(idx_c)
        if n == 0:
            continue

        p = np.random.dirichlet([alpha] * num_clients)
        counts = (p * n).astype(int)
        diff = n - counts.sum()
        if diff > 0:
            order = np.argsort(p)[::-1]
            for j in order[:diff]:
                counts[j] += 1

        start = 0
        for client_id in range(num_clients):
            cnt = int(counts[client_id])
            if cnt <= 0:
                continue
            end = start + cnt
            client_indices[client_id].extend(idx_c[start:end].tolist())
            start = end

        if start < n:
            kmax = int(np.argmax(p))
            client_indices[kmax].extend(idx_c[start:n].tolist())

    return [np.array(ci, dtype=int) for ci in client_indices]


def build_vendor_client_mapping(
    vendors: List[str],
    satellites_per_vendor: int,
) -> Dict[str, List[int]]:
    """
    Assign client IDs to vendors for logging / grouping.
    Returns a dict: vendor -> list of client_ids.
    """
    mapping: Dict[str, List[int]] = {}
    cid = 0
    for v in vendors:
        sats = []
        for _ in range(satellites_per_vendor):
            sats.append(cid)
            cid += 1
        mapping[v] = sats
    return mapping


def build_client_loaders(
    train_dataset,
    labels_full: np.ndarray,
    num_clients: int,
    split_mode: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[List[DataLoader], List[int]]:
    """
    Returns:
      - list of DataLoaders (one per client)
      - list of dataset sizes (one per client)
    """
    all_idx = np.arange(len(labels_full))
    # we assume train_dataset is already a subset of full dataset,
    # but we partition over indices 0..len(train_dataset)-1 for simplicity.

    # For simplicity here we assume train_dataset is a Subset or has contiguous 0..N indices.
    train_indices = np.arange(len(train_dataset))

    if split_mode == "iid":
        parts = iid_partition_indices(train_indices, num_clients)
    elif split_mode == "noniid":
        parts = dirichlet_partition_indices(train_indices, labels_full[train_indices], num_clients)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode} (use 'iid' or 'noniid')")

    loaders = []
    sizes = []
    for idx in parts:
        ds = Subset(train_dataset, idx.tolist())
        loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=False,
            )
        )
        sizes.append(len(ds))

    return loaders, sizes
