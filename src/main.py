from __future__ import annotations

import argparse

from src.scripts import build_dataset, compare_models, run_eda, train_all, train_baselines, train_model, train_pinn


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RP-3 PINN project entrypoint.")
    parser.add_argument("command", choices=["build_dataset", "run_eda", "train_baselines", "train_pinn", "train_model", "train_all", "compare_models"])
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0], *remaining]
    if args.command == "build_dataset":
        build_dataset.main()
    elif args.command == "run_eda":
        run_eda.main()
    elif args.command == "train_baselines":
        train_baselines.main()
    elif args.command == "train_pinn":
        train_pinn.main()
    elif args.command == "train_model":
        train_model.main()
    elif args.command == "train_all":
        train_all.main()
    elif args.command == "compare_models":
        compare_models.main()


if __name__ == "__main__":
    main()
