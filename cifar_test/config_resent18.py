"""Single source of truth for the CIFAR-10 ResNet-18 experiment.

Edit this file before moving the project to another machine.  The fixed
warm-up artifact is identified by ``resnet18_cifar10_sn40_warmup50``; change
that identifier whenever settings that affect warm-up training are changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: Path = PROJECT_ROOT / "data" / "cifar10"
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
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    model_id: str = "resnet18_cifar10_sn40_warmup50"
    epochs: int = 50
    freeze_epochs: int = 0
    batch_size: int = 128
    eval_batch_size: int = 1024 #256
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = None
    scheduler: str = "step_halfway"
    lr_decay_fraction: float = 0.5
    lr_decay_factor: float = 0.1
    deployment_checkpoint: str = "best"
    min_noisy_validation_accuracy: float = 0.0


@dataclass(frozen=True, slots=True)
class KNNConfig:
    k: int = 10
    temperature: float = 0.5
    query_chunk_size: int = 4096 #2_048
    reference_chunk_size: int = 65536 #32_768
    correction_chunk_size: int = 16_384


@dataclass(frozen=True, slots=True)
class RLConfig:
    epochs: int = 500
    trajectory_length: int = 10
    cleaning_trajectory_length: int = 25
    initial_state_randomization_rate: float = 0.10
    feature_batch_size: int = 1024 #128
    update_mode: str = "full"
    update_batch_size: int = 512 #128

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
    deployment_checkpoint: str = "last"


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
    label_smoothing: float = 0.0


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_channels_last: bool = True
    cudnn_benchmark: bool = True
    final_test_batch_size: int = 1024 #256
    overwrite_noise: bool = False
    overwrite_warmup: bool = False
    overwrite_rl: bool = False
    overwrite_finetune: bool = False
    overwrite_final_test: bool = False


@dataclass(frozen=True, slots=True)
class ResNet18CIFARConfig:
    output_root: Path = PROJECT_ROOT / "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

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
        return self.warmup_output_dir / f"{self.warmup.model_id}.pt"

    @property
    def rl_output_dir(self) -> Path:
        return self.output_root / (
            f"cifar10_rl_{self.warmup.model_id}_"
            f"{self.rl.update_mode}_noise{self.noise_tag}"
        )

    @property
    def finetune_output_dir(self) -> Path:
        return self.output_root / (
            f"cifar10_finetune_{self.warmup.model_id}_noise{self.noise_tag}"
        )

    @property
    def final_test_output_dir(self) -> Path:
        return self.output_root / (
            f"cifar10_final_test_{self.warmup.model_id}_noise{self.noise_tag}"
        )

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
        if self.rl.update_mode != "full":
            raise ValueError("This baseline requires full actor updates.")
        if self.warmup.deployment_checkpoint != "best":
            raise ValueError("Warm-up deployment must use the best checkpoint.")
        if self.rl.deployment_checkpoint != "last":
            raise ValueError("RL deployment must use the last checkpoint.")
        positive_values = (
            self.warmup.epochs,
            self.warmup.batch_size,
            self.rl.epochs,
            self.rl.trajectory_length,
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
