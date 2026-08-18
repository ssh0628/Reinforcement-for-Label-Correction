"""Single source of truth for the CIFAR-10 ResNet-18 experiment.

Edit this file before moving the project to another machine. In particular,
``DataConfig.root`` and ``OutputConfig.root`` may be changed to absolute paths
on the training machine. No output path is tied to the repository location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataConfig:
    # Change to an absolute path on the training machine if desired.
    root: Path = Path("data/cifar10")
    download: bool = True
    classes: tuple[int, ...] = tuple(range(10))
    train_samples: int = 50_000
    image_size: int = 32
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    noise_rate: float = 0.40
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "cifar_resnet18"
    pretrained: bool = False


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    model_id: str = "resnet18_cifar10_sn40_warmup50"
    epochs: int = 50
    batch_size: int = 128
    eval_batch_size: int = 1_024
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_decay_fraction: float = 0.5
    lr_decay_factor: float = 0.1
    min_noisy_validation_accuracy: float = 0.0


@dataclass(frozen=True, slots=True)
class KNNConfig:
    k: int = 10
    temperature: float = 0.5
    query_chunk_size: int = 4_096
    reference_chunk_size: int = 65_536
    correction_chunk_size: int = 16_384


@dataclass(frozen=True, slots=True)
class RLConfig:
    epochs: int = 500
    trajectory_length: int = 10
    cleaning_trajectory_length: int = 25
    initial_state_randomization_rate: float = 0.10
    feature_batch_size: int = 1_024
    update_mode: str = "full"  # "full" or "subset"
    subset_size: int = 5_000
    update_batch_size: int = 512

    actor_optimizer: str = "sgd"
    actor_learning_rate: float = 1e-2
    actor_momentum: float = 0.9
    actor_weight_decay: float = 5e-4

    critic_optimizer: str = "sgd"
    critic_learning_rate: float = 1e-2
    critic_momentum: float = 0.9
    critic_weight_decay: float = 5e-4
    critic_num_bins: int = 100

    discount_factor: float = 0.9
    reward_nla_weight: float = 0.5
    lr_decay_fraction: float = 0.5
    lr_decay_factor: float = 0.1


@dataclass(frozen=True, slots=True)
class OutputConfig:
    # This is intentionally not based on PROJECT_ROOT. Set an absolute path
    # here when running on another machine, e.g. Path("/mnt/results/rlnlc").
    root: Path = Path("outputs")

    warmup_checkpoint_name: str = "resnet18_cifar10_sn40_warmup50.pt"
    actor_best_checkpoint_name: str = "actor_best.pt"
    actor_last_checkpoint_name: str = "actor_last.pt"
    critic_best_checkpoint_name: str = "critic_best.pt"
    critic_last_checkpoint_name: str = "critic_last.pt"
    corrected_labels_name: str = "train_corrected_labels.npy"
    finetune_checkpoint_name: str = "finetune_last.pt"


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_decay_fraction: float = 0.5
    lr_decay_factor: float = 0.1


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_channels_last: bool = True
    cudnn_benchmark: bool = True
    final_test_batch_size: int = 1_024
    overwrite_noise: bool = False
    overwrite_warmup: bool = False
    overwrite_rl: bool = False
    overwrite_finetune: bool = False
    overwrite_final_test: bool = False


@dataclass(frozen=True, slots=True)
class ResNet18CIFARConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def output_root(self) -> Path:
        return self.output.root.expanduser()

    @property
    def noise_tag(self) -> int:
        return round(self.data.noise_rate * 100)

    @property
    def noise_output_dir(self) -> Path:
        return (
            self.output_root
            / "cifar10_shared"
            / f"noise_{self.noise_tag}_seed{self.data.seed}"
        )

    @property
    def warmup_output_dir(self) -> Path:
        return (
            self.output_root
            / "cifar10_shared"
            / "warmup"
            / self.warmup.model_id
        )

    @property
    def warmup_checkpoint_path(self) -> Path:
        return self.warmup_output_dir / self.output.warmup_checkpoint_name

    @property
    def rl_output_dir(self) -> Path:
        return self.output_root / (
            f"cifar10_rl_{self.warmup.model_id}_"
            f"{self.rl.update_mode}_noise{self.noise_tag}"
        )

    @property
    def actor_update_samples(self) -> int:
        if self.rl.update_mode == "full":
            return self.data.train_samples
        return self.rl.subset_size

    @property
    def actor_best_checkpoint_path(self) -> Path:
        return self.rl_output_dir / self.output.actor_best_checkpoint_name

    @property
    def actor_last_checkpoint_path(self) -> Path:
        return self.rl_output_dir / self.output.actor_last_checkpoint_name

    @property
    def critic_best_checkpoint_path(self) -> Path:
        return self.rl_output_dir / self.output.critic_best_checkpoint_name

    @property
    def critic_last_checkpoint_path(self) -> Path:
        return self.rl_output_dir / self.output.critic_last_checkpoint_name

    @property
    def corrected_labels_path(self) -> Path:
        return self.rl_output_dir / self.output.corrected_labels_name

    @property
    def finetune_output_dir(self) -> Path:
        return self.output_root / (
            f"cifar10_finetune_{self.warmup.model_id}_"
            f"{self.rl.update_mode}_noise{self.noise_tag}"
        )

    @property
    def final_test_output_dir(self) -> Path:
        return self.output_root / (
            f"cifar10_final_test_{self.warmup.model_id}_"
            f"{self.rl.update_mode}_noise{self.noise_tag}"
        )

    @property
    def finetune_checkpoint_path(self) -> Path:
        return self.finetune_output_dir / self.output.finetune_checkpoint_name

    def validate(self) -> None:
        if self.model.pretrained:
            raise ValueError("The paper-style ResNet-18 must not be pretrained.")
        if self.data.classes != tuple(range(10)):
            raise ValueError("CIFAR-10 classes must be 0 through 9.")
        if not 0 <= self.data.noise_rate < 1:
            raise ValueError("noise_rate must be in [0, 1).")
        if self.data.train_samples <= 0 or self.data.image_size <= 0:
            raise ValueError("Dataset sizes must be positive.")
        if not self.warmup.model_id:
            raise ValueError("warmup.model_id must not be empty.")
        if self.warmup.optimizer != "sgd" or self.rl.actor_optimizer != "sgd":
            raise ValueError("This paper-style baseline requires SGD.")
        if self.rl.critic_optimizer != "sgd" or self.finetune.optimizer != "sgd":
            raise ValueError("Critic and fine-tuning optimizers must be SGD.")
        if self.rl.update_mode not in {"full", "subset"}:
            raise ValueError("rl.update_mode must be 'full' or 'subset'.")
        if not 0 < self.rl.subset_size <= self.data.train_samples:
            raise ValueError(
                "rl.subset_size must be in [1, data.train_samples]."
            )
        if self.rl.update_batch_size > self.actor_update_samples:
            raise ValueError(
                "rl.update_batch_size cannot exceed actor_update_samples."
            )
        if self.knn.k >= self.data.train_samples:
            raise ValueError("knn.k must be smaller than data.train_samples.")
        output_names = (
            self.output.warmup_checkpoint_name,
            self.output.actor_best_checkpoint_name,
            self.output.actor_last_checkpoint_name,
            self.output.critic_best_checkpoint_name,
            self.output.critic_last_checkpoint_name,
            self.output.corrected_labels_name,
            self.output.finetune_checkpoint_name,
        )
        if not all(name and Path(name).name == name for name in output_names):
            raise ValueError("Output artifact names must be non-empty filenames.")
        positive_values = (
            self.warmup.epochs,
            self.warmup.batch_size,
            self.rl.epochs,
            self.rl.trajectory_length,
            self.rl.feature_batch_size,
            self.rl.update_batch_size,
            self.knn.k,
            self.knn.temperature,
            self.finetune.epochs,
            self.finetune.batch_size,
            self.runtime.final_test_batch_size,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError(
                "Training counts and positive hyperparameters must be positive."
            )


CONFIG = ResNet18CIFARConfig()
CONFIG.validate()
