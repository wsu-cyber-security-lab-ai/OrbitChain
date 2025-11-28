import os
import csv
from typing import Dict, List, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import DEVICE
from .utils import set_seed
from .partition import build_vendor_client_mapping, build_client_loaders
from .datasets import _get_labels_array, load_dataset
from .models import get_model_builder
from .datasets import load_dataset
from .models import get_model_builder
from .partition import build_client_loaders
from .visibility import build_visible_clients_schedule



# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


# -------------------------------
# CSV + Directory Helpers
# -------------------------------

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_csv(path, header, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# -------------------------------
# State functions
# -------------------------------

def get_state_dict(m: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def set_state_dict(m: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    m.load_state_dict(sd, strict=True)


# -------------------------------
# Evaluation
# -------------------------------

@torch.no_grad()
def evaluate_model(m: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    m.eval().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = m(x)
        loss = crit(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


# -------------------------------
# Local Training
# -------------------------------

def local_train(
    base_sd: Dict[str, torch.Tensor],
    model_builder: Callable[[], nn.Module],
    train_loader: DataLoader,
    lr: float = 1e-3,
    epochs: int = 1,
    clip_norm: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], float, float]:

    m = model_builder()
    set_state_dict(m, base_sd)
    m.train().to(DEVICE)

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = m(x)
            loss = crit(out, y)
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip_norm)
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_samples += x.size(0)

    if total_samples == 0:
        return get_state_dict(m), 0.0, 0.0

    local_loss = total_loss / total_samples
    local_acc = total_correct / total_samples
    return get_state_dict(m), local_loss, local_acc


# -------------------------------
# FedAvg
# -------------------------------

def fedavg_weighted(
    updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:

    total_w = sum(w for _, w in updates)
    keys = updates[0][0].keys()
    agg = {k: torch.zeros_like(updates[0][0][k]) for k in keys}
    for sd, w in updates:
        alpha = w / (total_w + 1e-12)
        for k in keys:
            agg[k] += sd[k] * alpha
    return agg


# ------------------------------------------------------
# Main FL Runner (TLE visibility + CSV + plots + TB)
# ------------------------------------------------------

def run_federated(
    dataset_name: str,
    split_mode: str,
    vendors: List[str],
    satellites_per_vendor: int,
    rounds: int = 5,
    batch_size: int = 32,
    ucm_root: str = "./data/UCMerced_LandUse/Images",
    lr: float = 1e-3,
    local_epochs: int = 1,
    clip_norm: float = 1.0,
    use_real_tle_visibility: bool = True,
    cycle_minutes: int = 20,
    output_dir: str = "./runs/default_run",
    tensorboard: bool = True,
):
    """
    High-level FL run with:
      • dataset_name: "mnist", "eurosat", "ucmerced"
      • split_mode: "iid" or "noniid"
      • vendors: e.g. ["Starlink", "OneWeb", "Kuiper"]
      • satellites_per_vendor: clients per vendor
      • use_real_tle_visibility: if True, try Skyfield+TLE; else synthetic
      • cycle_minutes: used by visibility schedule
      • tensorboard: enable/disable SummaryWriter logging
    """

    set_seed()
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "plots"))

    # TensorBoard
    writer: Optional["SummaryWriter"] = None
    if tensorboard and _HAS_TB:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
        print(f"[TB] Logging to {os.path.join(output_dir, 'tb')}")
    elif tensorboard:
        print("[TB] torch.utils.tensorboard not available → skipping TB logging.")

    # -------------------------
    # Load dataset
    # -------------------------
    train_ds, test_ds, num_classes = load_dataset(dataset_name, ucm_root=ucm_root)
    labels_full = _get_labels_array(train_ds)

    # -------------------------
    # Partition clients
    # -------------------------
    num_clients = len(vendors) * satellites_per_vendor
    vendor_map = build_vendor_client_mapping(vendors, satellites_per_vendor)
    print("Vendor → client_ids mapping:", vendor_map)

    client_loaders, client_sizes = build_client_loaders(
        train_dataset=train_ds,
        labels_full=labels_full,
        num_clients=num_clients,
        split_mode=split_mode,
        batch_size=batch_size,
    )

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # -------------------------
    # Build visibility schedule (per satellite / client)
    # -------------------------
    if use_real_tle_visibility:
        visible_clients_per_round = build_visible_clients_schedule(
            vendor_map=vendor_map,
            num_rounds=rounds,
            cycle_minutes=cycle_minutes,
        )
        print("[visibility] Using real TLE-based (or fallback) satellite schedule.")
    else:
        from .visibility import _build_synthetic_visible_clients
        visible_clients_per_round = _build_synthetic_visible_clients(
            vendor_map, num_rounds=rounds, p_visible=0.7, seed=42
        )
        print("[visibility] Using synthetic satellite visibility (no TLE).")

    # -------------------------
    # Init model
    # -------------------------
    model_builder = get_model_builder(dataset_name, num_classes)
    global_model = model_builder()
    global_sd = get_state_dict(global_model)

    # -------------------------
    # CSV Logs
    # -------------------------
    global_rows = []          # round-level
    local_rows_all_rounds = []  # client-level

    # -------------------------
    # Training Loop
    # -------------------------
    for r in range(rounds):
        round_id = r + 1
        print(f"\n=== Round {round_id}/{rounds} ===")

        active_clients = visible_clients_per_round[r]
        print(f"Active clients this round: {active_clients}")

        updates = []
        round_local_rows = []

        for cid in active_clients:
            local_sd, local_loss, local_acc = local_train(
                base_sd=global_sd,
                model_builder=model_builder,
                train_loader=client_loaders[cid],
                lr=lr,
                epochs=local_epochs,
                clip_norm=clip_norm,
            )
            updates.append((local_sd, client_sizes[cid]))

            # Which vendor?
            vendor_name = [v for v, ids in vendor_map.items() if cid in ids][0]
            round_local_rows.append([
                round_id,
                cid,
                vendor_name,
                client_sizes[cid],
                local_loss,
                local_acc,
            ])

        # Save round local CSV
        write_csv(
            os.path.join(output_dir, f"local_accuracy_round{round_id}.csv"),
            ["round", "client_id", "vendor", "size", "local_loss", "local_acc"],
            round_local_rows,
        )
        local_rows_all_rounds.extend(round_local_rows)

        # FedAvg
        global_sd = fedavg_weighted(updates)
        set_state_dict(global_model, global_sd)

        # Evaluate global
        g_loss, g_acc = evaluate_model(global_model, test_loader)
        print(f"[Round {round_id}] Global Acc = {g_acc*100:.2f}%  Loss = {g_loss:.4f}")

        # Global row
        global_rows.append([
            round_id,
            g_loss,
            g_acc,
            len(active_clients),
        ])

        # TensorBoard: global
        if writer is not None:
            writer.add_scalar("global/loss", g_loss, round_id)
            writer.add_scalar("global/acc", g_acc, round_id)
            writer.add_scalar("global/active_clients", len(active_clients), round_id)

            # Vendor-wise mean local acc for this round
            vendors_in_round = {}
            for (_, _, vname, _, _, lacc) in round_local_rows:
                vendors_in_round.setdefault(vname, []).append(lacc)
            for vname, accs in vendors_in_round.items():
                writer.add_scalar(
                    f"vendor/{vname}/mean_local_acc",
                    float(sum(accs) / max(1, len(accs))),
                    round_id,
                )

    # Save global history CSV
    write_csv(
        os.path.join(output_dir, "global_history.csv"),
        ["round", "global_loss", "global_acc", "active_clients"],
        global_rows,
    )

    # Save full local CSV
    write_csv(
        os.path.join(output_dir, "local_accuracy_all_rounds.csv"),
        ["round", "client_id", "vendor", "size", "local_loss", "local_acc"],
        local_rows_all_rounds,
    )

    # -------------------------
    # Matplotlib plots
    # -------------------------
    import matplotlib.pyplot as plt

    rounds_list = [row[0] for row in global_rows]
    loss_list = [row[1] for row in global_rows]
    acc_list = [row[2] for row in global_rows]

    # Global accuracy curve
    plt.figure(figsize=(8, 4))
    plt.plot(rounds_list, [a * 100 for a in acc_list], marker="o")
    plt.title("Global Accuracy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "global_accuracy.png"))
    plt.close()

    # Global loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(rounds_list, loss_list, marker="o", color="tab:red")
    plt.title("Global Loss vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "global_loss.png"))
    plt.close()

    # Vendor-wise mean local accuracy curves
    # Build dict: vendor -> { round -> [accs] }
    vendor_round_accs: Dict[str, Dict[int, List[float]]] = {}
    for (rnd, cid, vname, size, lloc, lacc) in local_rows_all_rounds:
        vendor_round_accs.setdefault(vname, {}).setdefault(rnd, []).append(lacc)

    plt.figure(figsize=(8, 4))
    for vname, round_dict in vendor_round_accs.items():
        rds = sorted(round_dict.keys())
        means = [
            float(sum(round_dict[rid]) / max(1, len(round_dict[rid])))
            for rid in rds
        ]
        plt.plot(rds, [m * 100 for m in means], marker="o", label=vname)

    plt.title("Vendor-wise Mean Local Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Local Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "vendor_mean_local_accuracy.png"))
    plt.close()

    if writer is not None:
        writer.close()

    print(f"\n✔ Logs saved to: {output_dir}")
    print(f"✔ Plots saved to: {os.path.join(output_dir, 'plots')}")
    if writer is not None:
        print(f"✔ TensorBoard logs: {os.path.join(output_dir, 'tb')}")
