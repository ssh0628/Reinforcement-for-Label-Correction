"""Single source of truth for the CIFAR-10 ResNet-18 experiment.

The default project location targets the H100 machine. Change ``PROJECT_ROOT``
once if the repository is moved; data and output paths follow automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path("/root/project/rlnlc")


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
    initial_state_randomization_rate: float = 0.10
    feature_batch_size: int = 1_024
    update_mode: str = "full"  # "full" or "subset"
    subset_size: int = 5_000
    # Direct query microbatch (memory control only). Each forward can contain
    # up to update_batch_size * (k + 1) query/neighbor image occurrences.
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
class CorrectionConfig:
    trajectory_length: int = 25


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Compact experiment layout rooted at ``cifar_output/expN``."""

    root: Path = PROJECT_ROOT / "cifar_output"
    experiment_name: str = "exp1"

    warmup_checkpoint_name: str = "warmup.pt"
    actor_best_checkpoint_name: str = "actor_best.pt"
    actor_last_checkpoint_name: str = "actor_last.pt"
    critic_best_checkpoint_name: str = "critic_best.pt"
    critic_last_checkpoint_name: str = "critic_last.pt"
    corrected_labels_name: str = "train_corrected_labels.npy"
    finetune_checkpoint_name: str = "finetune_last.pt"


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    initialization: str = "last_actor"  # warmup, best_actor, or last_actor
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
    evaluate_batch_size: int = 1_024
    overwrite_noise: bool = False
    overwrite_warmup: bool = False
    overwrite_rl: bool = False
    overwrite_correction: bool = False
    overwrite_finetune: bool = False
    overwrite_evaluate: bool = False


@dataclass(frozen=True, slots=True)
class ResNet18CIFARConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def data_root(self) -> Path:
        return self.data.root.expanduser()

    @property
    def output_root(self) -> Path:
        return self.output.root.expanduser()

    @property
    def noise_tag(self) -> int:
        return round(self.data.noise_rate * 100)

    @property
    def noise_output_dir(self) -> Path:
        return self.data_root / (
            f"cifar10_noise_{self.noise_tag}_seed{self.data.seed}"
        )

    @property
    def warmup_output_dir(self) -> Path:
        return self.experiment_output_dir / "warmup"

    @property
    def warmup_log_dir(self) -> Path:
        return self.warmup_output_dir / "logs"

    @property
    def warmup_model_dir(self) -> Path:
        return self.warmup_output_dir / "model"

    @property
    def warmup_checkpoint_path(self) -> Path:
        return self.warmup_model_dir / self.output.warmup_checkpoint_name

    @property
    def experiment_id(self) -> str:
        return self.output.experiment_name

    @property
    def experiment_output_dir(self) -> Path:
        return self.output_root / self.experiment_id

    @property
    def mode_output_dir(self) -> Path:
        return self.experiment_output_dir / self.rl.update_mode

    @property
    def log_output_dir(self) -> Path:
        return self.mode_output_dir / "logs"

    @property
    def rl_output_dir(self) -> Path:
        return self.log_output_dir / "rl"

    @property
    def rl_model_dir(self) -> Path:
        return self.mode_output_dir / "model_rl"

    @property
    def actor_update_samples(self) -> int:
        if self.rl.update_mode == "full":
            return self.data.train_samples
        return self.rl.subset_size

    @property
    def actor_best_checkpoint_path(self) -> Path:
        return self.rl_model_dir / self.output.actor_best_checkpoint_name

    @property
    def actor_last_checkpoint_path(self) -> Path:
        return self.rl_model_dir / self.output.actor_last_checkpoint_name

    @property
    def critic_best_checkpoint_path(self) -> Path:
        return self.rl_model_dir / self.output.critic_best_checkpoint_name

    @property
    def critic_last_checkpoint_path(self) -> Path:
        return self.rl_model_dir / self.output.critic_last_checkpoint_name

    @property
    def correction_actor_checkpoint_path(self) -> Path:
        return self.actor_last_checkpoint_path

    @property
    def correction_output_dir(self) -> Path:
        return self.log_output_dir / "correction"

    @property
    def corrected_labels_path(self) -> Path:
        return self.rl_model_dir / self.output.corrected_labels_name

    @property
    def finetune_initial_checkpoint_path(self) -> Path:
        sources = {
            "warmup": self.warmup_checkpoint_path,
            "best_actor": self.actor_best_checkpoint_path,
            "last_actor": self.actor_last_checkpoint_path,
        }
        return sources[self.finetune.initialization]

    @property
    def finetune_output_dir(self) -> Path:
        return self.log_output_dir / "finetune"

    @property
    def finetune_model_dir(self) -> Path:
        return self.mode_output_dir / "model_finetune"

    @property
    def evaluate_output_dir(self) -> Path:
        return self.log_output_dir / "evaluate"

    @property
    def finetune_checkpoint_path(self) -> Path:
        return self.finetune_model_dir / self.output.finetune_checkpoint_name

    def validate(self) -> None:
        if self.model.pretrained:
            raise ValueError("The paper-style ResNet-18 must not be pretrained.")
        if not self.data_root.is_absolute() or not self.output_root.is_absolute():
            raise ValueError("Data and output roots must be absolute paths.")
        if self.data.classes != tuple(range(10)):
            raise ValueError("CIFAR-10 classes must be 0 through 9.")
        if not 0 <= self.data.noise_rate < 1:
            raise ValueError("noise_rate must be in [0, 1).")
        if self.data.train_samples != 50_000 or self.data.image_size != 32:
            raise ValueError(
                "The CIFAR-10 baseline requires 50,000 native 32x32 images."
            )
        if len(self.data.mean) != 3 or len(self.data.std) != 3:
            raise ValueError("CIFAR-10 mean and std must contain three values.")
        if any(value <= 0 for value in self.data.std):
            raise ValueError("CIFAR-10 standard deviations must be positive.")
        if not self.warmup.model_id:
            raise ValueError("warmup.model_id must not be empty.")
        if (
            self.warmup.optimizer.lower() != "sgd"
            or self.rl.actor_optimizer.lower() != "sgd"
        ):
            raise ValueError("This paper-style baseline requires SGD.")
        if (
            self.rl.critic_optimizer.lower() != "sgd"
            or self.finetune.optimizer.lower() != "sgd"
        ):
            raise ValueError("Critic and fine-tuning optimizers must be SGD.")
        if self.rl.update_mode not in {"full", "subset"}:
            raise ValueError("rl.update_mode must be 'full' or 'subset'.")
        if self.finetune.initialization not in {
            "warmup",
            "best_actor",
            "last_actor",
        }:
            raise ValueError(
                "finetune.initialization must be warmup, best_actor, or "
                "last_actor."
            )
        if not 0 < self.rl.subset_size <= self.data.train_samples:
            raise ValueError(
                "rl.subset_size must be in [1, data.train_samples]."
            )
        if self.rl.update_batch_size > self.actor_update_samples:
            raise ValueError(
                "rl.update_batch_size cannot exceed actor_update_samples."
            )
        if not 0 < self.rl.initial_state_randomization_rate < 1:
            raise ValueError(
                "rl.initial_state_randomization_rate must be in (0, 1)."
            )
        if not 0 <= self.rl.discount_factor <= 1:
            raise ValueError("rl.discount_factor must be in [0, 1].")
        if self.rl.reward_nla_weight < 0 or self.rl.critic_num_bins < 2:
            raise ValueError(
                "RL reward weight must be non-negative and critic bins at "
                "least two."
            )
        momentums = (
            self.warmup.momentum,
            self.rl.actor_momentum,
            self.rl.critic_momentum,
            self.finetune.momentum,
        )
        if any(value < 0 for value in momentums):
            raise ValueError("SGD momentums must be non-negative.")
        if self.knn.k >= self.data.train_samples:
            raise ValueError("knn.k must be smaller than data.train_samples.")
        if (
            self.knn.query_chunk_size <= 0
            or self.knn.reference_chunk_size <= 0
            or self.knn.correction_chunk_size <= 0
        ):
            raise ValueError("KNN and correction chunk sizes must be positive.")
        learning_rates = (
            self.warmup.learning_rate,
            self.rl.actor_learning_rate,
            self.rl.critic_learning_rate,
            self.finetune.learning_rate,
        )
        if any(value <= 0 for value in learning_rates):
            raise ValueError("All learning rates must be positive.")
        weight_decays = (
            self.warmup.weight_decay,
            self.rl.actor_weight_decay,
            self.rl.critic_weight_decay,
            self.finetune.weight_decay,
        )
        if any(value < 0 for value in weight_decays):
            raise ValueError("Weight decays must be non-negative.")
        schedule_values = (
            (self.warmup.lr_decay_fraction, self.warmup.lr_decay_factor),
            (self.rl.lr_decay_fraction, self.rl.lr_decay_factor),
            (self.finetune.lr_decay_fraction, self.finetune.lr_decay_factor),
        )
        if any(
            not 0 < fraction <= 1 or not 0 < factor <= 1
            for fraction, factor in schedule_values
        ):
            raise ValueError(
                "LR decay fractions and factors must be in (0, 1]."
            )
        if self.runtime.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("runtime.amp_dtype must be float16 or bfloat16.")
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
        rl_checkpoint_names = (
            self.output.actor_best_checkpoint_name,
            self.output.actor_last_checkpoint_name,
            self.output.critic_best_checkpoint_name,
            self.output.critic_last_checkpoint_name,
        )
        if len(set(rl_checkpoint_names)) != len(rl_checkpoint_names):
            raise ValueError("RL checkpoint filenames must be distinct.")
        if (
            not self.output.experiment_name
            or Path(self.output.experiment_name).name
            != self.output.experiment_name
            or self.output.experiment_name in {".", ".."}
        ):
            raise ValueError(
                "output.experiment_name must be one non-empty path component."
            )
        positive_values = (
            self.warmup.epochs,
            self.warmup.batch_size,
            self.warmup.eval_batch_size,
            self.rl.epochs,
            self.rl.trajectory_length,
            self.correction.trajectory_length,
            self.rl.feature_batch_size,
            self.rl.update_batch_size,
            self.knn.k,
            self.knn.temperature,
            self.finetune.epochs,
            self.finetune.batch_size,
            self.runtime.evaluate_batch_size,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError(
                "Training counts and positive hyperparameters must be positive."
            )


CONFIG = ResNet18CIFARConfig()
CONFIG.validate()
