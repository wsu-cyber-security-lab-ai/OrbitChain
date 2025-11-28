import argparse
from satfl.fl_core import run_federated




def parse_args():
    p = argparse.ArgumentParser(
        description="Unified FL runner for MNIST / EuroSAT / UC Merced "
                    "with IID / non-IID and multi-vendor support."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "eurosat", "ucmerced"],
        help="Dataset name.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="iid",
        choices=["iid", "noniid"],
        help="Data split mode: iid or noniid (Dirichlet).",
    )
    p.add_argument(
        "--vendors",
        type=str,
        default="Starlink",
        help="Comma-separated list of vendor names, e.g. 'Starlink,OneWeb,Kuiper'.",
    )
    p.add_argument(
        "--sats-per-vendor",
        type=int,
        default=3,
        help="Number of clients (satellites) per vendor.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of FL rounds.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Local batch size.",
    )
    p.add_argument(
        "--ucm-root",
        type=str,
        default="./data/UCMerced_LandUse/Images",
        help="Root path to UC Merced 'Images' folder (for ucmerced dataset).",
    )
    p.add_argument("--lr", type=float, default=1e-3, help="Local learning rate.")
    p.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round.")
    p.add_argument("--clip-norm", type=float, default=1.0, help="Grad clipping norm.")

    return p.parse_args()


def main():
    args = parse_args()
    vendors = [v.strip() for v in args.vendors.split(",") if v.strip()]
# Add this near main()

    visibility_example = {
        "Starlink": [1,1,0,1,0],   # visible rounds 1,2,4
        "OneWeb":   [1,0,1,0,1],   # alternating
        "Kuiper":   [1,1,1,1,1],   # always visible
    }

    run_federated(
        dataset_name=args.dataset,
        split_mode=args.split,
        vendors=vendors,
        satellites_per_vendor=args.sats_per_vendor,
        rounds=args.rounds,
        batch_size=args.batch_size,
        ucm_root=args.ucm_root,
        lr=args.lr,
        local_epochs=args.local_epochs,
        clip_norm=args.clip_norm,
        use_real_tle_visibility=True,    #  Skyfield + TLE if available
        cycle_minutes=20,
        output_dir=f"./runs/{args.dataset}_{args.split}",
        tensorboard=True,
    )



if __name__ == "__main__":
    main()
