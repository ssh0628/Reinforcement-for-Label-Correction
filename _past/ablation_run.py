"""Run three subset Critic ablations through RL, correction, fine-tuning, and evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(
    (path for path in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (path / "cifar_test" / "__init__.py").is_file()),
    None,
)
if ROOT is None:
    raise RuntimeError("Could not locate the project root containing cifar_test/__init__.py.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    experiment_name: str
    optimizer: str
    terminal: bool
    horizon: bool
    hidden_dims: tuple[int, ...]


VARIANTS = (
    Variant("adam", "exp2", "adam", False, False, (128, 64)),
    Variant("terminal_horizon", "exp3", "sgd", True, True, (128, 64)),
    Variant("deep_mlp", "exp4", "sgd", False, False, (256, 128, 64)),
    Variant("terminal_horizon_deep_mlp", "exp5", "sgd", True, True, (256, 128, 64)),
    Variant("adam_terminal_horizon_deep_mlp", "exp6", "adam", True, True, (256, 128, 64)),
)
WARMUP_SOURCE_EXPERIMENT = "exp1"


def build_config(variant: Variant, epochs: int):
    from cifar_test.setting import config as config_module

    base = config_module.CONFIG
    configured = replace(
        base,
        rl=replace(
            base.rl,
            epochs=epochs,
            update_mode="subset",
            actor_step_mode="trajectory",
            update_batch_size=1_024,
            critic_optimizer=variant.optimizer,
            use_terminal_critic_update=variant.terminal,
            use_remaining_horizon=variant.horizon,
            critic_hidden_dims=variant.hidden_dims,
        ),
        output=replace(base.output, experiment_name=variant.experiment_name),
        finetune=replace(
            base.finetune,
            corrected_label_source="rl",
            initialization="last_actor",
            evaluation_checkpoint="accuracy",
        ),
        runtime=replace(
            base.runtime,
            overwrite_rl=False,
            overwrite_correction=False,
            overwrite_finetune=False,
            overwrite_evaluate=False,
        ),
    )
    configured.validate()
    return config_module, base, configured


def share_warmup(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Shared warm-up checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Warm-up destination already points elsewhere: {destination}")
        return
    destination.symlink_to(source.resolve())


def shared_warmup(base) -> tuple[Path, str]:
    import torch

    source = (
        base.output_root
        / WARMUP_SOURCE_EXPERIMENT
        / "warmup"
        / "model"
        / base.output.warmup_checkpoint_name
    )
    if not source.is_file():
        raise FileNotFoundError(f"Shared warm-up checkpoint not found: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model_id = str(checkpoint.get("warmup_model_id", "")) if isinstance(checkpoint, dict) else ""
    if not model_id:
        raise ValueError(f"Warm-up checkpoint has no warmup_model_id: {source}")
    return source, model_id


def run_variant(name: str, *, epochs: int, dry_run: bool, rl_only: bool) -> None:
    variant = next(item for item in VARIANTS if item.name == name)
    config_module, base, configured = build_config(variant, epochs)
    print(
        f"\n[VARIANT] {variant.name}\n"
        f"  output={configured.experiment_output_dir}\n"
        f"  epochs={configured.rl.epochs} subset={configured.rl.subset_size} "
        f"critic={variant.optimizer}:{configured.rl.effective_critic_options[0]:g}\n"
        f"  terminal={variant.terminal} horizon={variant.horizon} hidden={variant.hidden_dims}\n"
        "  correction=last_actor finetune=last_actor evaluate=best_accuracy",
        flush=True,
    )
    if dry_run:
        return

    warmup_source, warmup_model_id = shared_warmup(base)
    configured = replace(configured, warmup=replace(configured.warmup, model_id=warmup_model_id))
    configured.validate()
    share_warmup(warmup_source, configured.warmup_checkpoint_path)
    config_module.CONFIG = configured

    rl_complete = configured.actor_last_checkpoint_path.is_file() and (
        configured.rl_output_dir / "run_summary.csv"
    ).is_file()
    if rl_complete:
        print(f"[SKIP] completed RL: {variant.name}", flush=True)
    else:
        from cifar_test.rl.run import main as run_rl

        run_rl()

    if rl_only:
        return
    correction_complete = configured.corrected_labels_path.is_file() and (
        configured.correction_output_dir / "cleaning_summary.csv"
    ).is_file()
    if correction_complete:
        print(f"[SKIP] completed correction: {variant.name}", flush=True)
    else:
        from cifar_test.evaluate.correction import run_with_file_logging as run_correction

        run_correction()

    finetune_complete = all(
        path.is_file()
        for path in (
            configured.finetune_best_accuracy_checkpoint_path,
            configured.finetune_best_loss_checkpoint_path,
            configured.finetune_last_checkpoint_path,
            configured.finetune_output_dir / "train.csv",
        )
    )
    if finetune_complete:
        print(f"[SKIP] completed fine-tuning: {variant.name}", flush=True)
    else:
        from cifar_test.evaluate.finetuning import run_with_file_logging as run_finetuning

        run_finetuning()

    evaluation_complete = (configured.evaluate_output_dir / "test.csv").is_file()
    if evaluation_complete:
        print(f"[SKIP] completed evaluation: {variant.name}", flush=True)
    else:
        from cifar_test.evaluate.final import run_with_file_logging as run_evaluation

        run_evaluation()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500, help="RL epochs per variant (default: 500).")
    parser.add_argument("--dry-run", action="store_true", help="Print all configurations without training.")
    parser.add_argument("--rl-only", action="store_true", help="Skip correction, fine-tuning, and evaluation.")
    parser.add_argument("--variant", choices=[item.name for item in VARIANTS], help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.variant:
        run_variant(
            args.variant,
            epochs=args.epochs,
            dry_run=args.dry_run,
            rl_only=args.rl_only,
        )
        return

    failures: list[str] = []
    for index, variant in enumerate(VARIANTS, start=1):
        print(f"\n[ABLATION {index}/{len(VARIANTS)}] {variant.name}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--variant",
            variant.name,
            "--epochs",
            str(args.epochs),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.rl_only:
            command.append("--rl-only")
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            failures.append(variant.name)
            print(f"[FAILED] {variant.name} (exit={result.returncode}); continuing.", flush=True)

    if failures:
        raise SystemExit(f"Failed variants: {', '.join(failures)}")
    print("\nAll Critic ablations completed.", flush=True)


if __name__ == "__main__":
    main()
