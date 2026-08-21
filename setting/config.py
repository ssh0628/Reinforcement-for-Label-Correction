from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = tuple(f"A{index}" for index in range(1, 8))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_MEAN_RGB = tuple(round(channel * 255) for channel in IMAGENET_MEAN)


@dataclass(frozen=True, slots=True)
class DataConfig:
    npy_dir: Path = Path("/absolute/path/to/npy")
    train_labels_override: Path | None = None
    class_names: tuple[str, ...] = CLASS_NAMES
    image_size: int = 224
    letterbox_fill: tuple[int, ...] = IMAGENET_MEAN_RGB
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_degrees: float = 15.0
    color_jitter: tuple[float, float, float, float] = (0.1, 0.1, 0.05, 0.02)

    def paths_file(self, split: str) -> Path:
        return self.npy_dir / f"{split}_paths.npy"

    def labels_file(self, split: str) -> Path:
        if split == "train" and self.train_labels_override is not None:
            return self.train_labels_override
        return self.npy_dir / f"{split}_labels.npy"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    drop_rate: float = 0.1
    drop_path_rate: float = 0.2


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 20
    backbone_freeze_epochs: int = 5

    optimizer_name: str = "adamw"
    lr_head: float = 1e-3
    lr_unfrozen: float = 3e-5
    weight_decay: float = 0.1
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8

    scheduler_name: str = "cosine_annealing"
    min_lr: float = 1e-6

    use_weighted_ce: bool = True
    label_smoothing: float = 0.0

    use_sqrt_sampler: bool = True
    sampler_replacement: bool = True
    sampler_num_samples: int | None = None

    @property
    def cosine_t_max(self) -> int:
        return self.epochs - self.backbone_freeze_epochs


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    warmup_batch_size: int = 256
    global_knn_feature_batch_size: int = 512
    rl_feature_batch_size: int = 512
    num_workers: int = 20
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    train_drop_last: bool = True


@dataclass(frozen=True, slots=True)
class BestCheckpoint:
    metric: str
    mode: Literal["max", "min"]
    filename: str


def _default_best_checkpoints() -> tuple[BestCheckpoint, ...]:
    return (
        BestCheckpoint("macro_f1", "max", "best_macro_f1.pt"),
        BestCheckpoint("balanced_accuracy", "max", "best_balanced_accuracy.pt"),
        BestCheckpoint("loss", "min", "best_loss.pt"),
    )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    best: tuple[BestCheckpoint, ...] = field(default_factory=_default_best_checkpoints)
    last_filename: str = "last.pt"


@dataclass(frozen=True, slots=True)
class GlobalKNNConfig:
    k: int = 10
    split: str = "train"
    checkpoint_path: Path = PROJECT_ROOT / "outputs" / "warmup" / "best_macro_f1.pt"
    output_dir: Path = PROJECT_ROOT / "outputs" / "global_knn"
    query_chunk_size: int = 8192
    reference_chunk_size: int = 65536
    cache_features_on_device: bool = True
    overwrite: bool = False

    @property
    def artifact_path(self) -> Path:
        return self.output_dir / f"{self.split}_global_knn_k{self.k}.pt"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    temperature: float = 0.5
    correction_chunk_size: int = 65536


@dataclass(frozen=True, slots=True)
class RewardConfig:
    nla_weight: float = 0.5


@dataclass(frozen=True, slots=True)
class RLTrainConfig:
    epochs: int = 20
    trajectory_length: int = 10
    discount_factor: float = 0.9
    critic_num_bins: int = 100
    initial_state_randomization_rate: float = 0.10

    actor_optimizer_name: str = "adamw"
    actor_lr: float = 3e-5
    actor_weight_decay: float = 0.1
    actor_adamw_betas: tuple[float, float] = (0.9, 0.999)
    actor_adamw_eps: float = 1e-8

    critic_optimizer_name: str = "sgd"
    critic_lr: float = 1e-2
    critic_momentum: float = 0.9
    critic_weight_decay: float = 5e-4

    scheduler_name: str = "step_halfway"
    lr_decay_fraction: float = 0.5
    lr_decay_factor: float = 0.1

    policy_update_mode: str = "subset"  # "full" or "subset"
    policy_update_subset_size: int = 10000
    policy_update_batch_size: int = 512

    checkpoint_interval: int = 1
    checkpoint_filename: str = "last.pt"
    resume_checkpoint_path: Path | None = None
    overwrite: bool = False
    output_dir: Path = PROJECT_ROOT / "outputs" / "rl"


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    trajectory_length: int = 25
    checkpoint_path: Path = PROJECT_ROOT / "outputs" / "rl" / "last.pt"
    output_dir: Path = PROJECT_ROOT / "outputs" / "cleaning"
    artifact_filename: str = "train_cleaning.pt"
    corrected_labels_filename: str = "train_corrected_labels.npy"
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "auto"
    use_amp: bool = True
    use_channels_last: bool = True
    cudnn_benchmark: bool = True
    seed: int = 0
    output_dir: Path = PROJECT_ROOT / "outputs" / "warmup"


@dataclass(frozen=True, slots=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    global_knn: GlobalKNNConfig = field(default_factory=GlobalKNNConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl_train: RLTrainConfig = field(default_factory=RLTrainConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def num_classes(self) -> int:
        return len(self.data.class_names)

    def validate(self) -> None:
        if self.data.class_names != CLASS_NAMES:
            raise ValueError(f"class_names must be {CLASS_NAMES}.")
        if not self.data.npy_dir.is_absolute():
            raise ValueError("data.npy_dir must be an absolute path.")
        if self.data.train_labels_override is not None and not self.data.train_labels_override.is_absolute():
            raise ValueError("data.train_labels_override must be None or an absolute path.")
        if len(self.data.letterbox_fill) != 3 or not all(
            0 <= channel <= 255 for channel in self.data.letterbox_fill
        ):
            raise ValueError("data.letterbox_fill must contain three values in [0, 255].")
        if self.data.image_size <= 0:
            raise ValueError("data.image_size must be positive.")
        if not 0 <= self.data.horizontal_flip_p <= 1:
            raise ValueError("data.horizontal_flip_p must be in [0, 1].")
        if not 0 <= self.data.vertical_flip_p <= 1:
            raise ValueError("data.vertical_flip_p must be in [0, 1].")
        if self.data.rotation_degrees < 0:
            raise ValueError("data.rotation_degrees must be non-negative.")
        if len(self.data.color_jitter) != 4 or any(value < 0 for value in self.data.color_jitter):
            raise ValueError("data.color_jitter must contain four non-negative values.")
        if self.data.color_jitter[3] > 0.5:
            raise ValueError("The color-jitter hue must not exceed 0.5.")
        if not 0 <= self.model.drop_rate < 1:
            raise ValueError("model.drop_rate must be in [0, 1).")
        if not 0 <= self.model.drop_path_rate < 1:
            raise ValueError("model.drop_path_rate must be in [0, 1).")
        if self.train.epochs <= 0:
            raise ValueError("train.epochs must be positive.")
        if not 0 <= self.train.backbone_freeze_epochs < self.train.epochs:
            raise ValueError("backbone_freeze_epochs must be in [0, epochs).")
        if self.train.optimizer_name.lower() != "adamw":
            raise ValueError("Only AdamW is supported by the warmup baseline.")
        if self.train.scheduler_name != "cosine_annealing":
            raise ValueError("Only cosine_annealing is supported by the warmup baseline.")
        if self.train.lr_head <= 0 or self.train.lr_unfrozen <= 0:
            raise ValueError("Warmup learning rates must be positive.")
        if self.train.weight_decay < 0:
            raise ValueError("train.weight_decay must be non-negative.")
        beta1, beta2 = self.train.adamw_betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("train.adamw_betas must be in [0, 1).")
        if self.train.adamw_eps <= 0:
            raise ValueError("train.adamw_eps must be positive.")
        if not 0 <= self.train.min_lr <= self.train.lr_unfrozen:
            raise ValueError("train.min_lr must be in [0, lr_unfrozen].")
        if not 0 <= self.train.label_smoothing <= 1:
            raise ValueError("train.label_smoothing must be in [0, 1].")
        if self.train.sampler_num_samples is not None and self.train.sampler_num_samples <= 0:
            raise ValueError("train.sampler_num_samples must be positive.")

        batch_sizes = {
            "warmup_batch_size": self.loader.warmup_batch_size,
            "global_knn_feature_batch_size": (self.loader.global_knn_feature_batch_size),
            "rl_feature_batch_size": self.loader.rl_feature_batch_size,
        }
        invalid_batch_sizes = [name for name, value in batch_sizes.items() if value <= 0]
        if invalid_batch_sizes:
            raise ValueError(f"Loader batch sizes must be positive: {invalid_batch_sizes}.")
        if self.loader.num_workers < 0:
            raise ValueError("loader.num_workers must be non-negative.")
        if self.loader.num_workers > 0 and self.loader.prefetch_factor <= 0:
            raise ValueError("loader.prefetch_factor must be positive when workers are used.")

        if self.global_knn.k <= 0:
            raise ValueError("global_knn.k must be positive.")
        if self.global_knn.split != "train":
            raise ValueError("global_knn.split must be train.")
        if self.global_knn.query_chunk_size <= 0 or self.global_knn.reference_chunk_size <= 0:
            raise ValueError("global_knn chunk sizes must be positive.")
        if not self.global_knn.checkpoint_path.is_absolute():
            raise ValueError("global_knn.checkpoint_path must be absolute.")
        if not self.global_knn.output_dir.is_absolute():
            raise ValueError("global_knn.output_dir must be absolute.")
        if self.policy.temperature <= 0:
            raise ValueError("policy.temperature must be positive.")
        if self.policy.correction_chunk_size <= 0:
            raise ValueError("policy.correction_chunk_size must be positive.")
        if self.reward.nla_weight < 0:
            raise ValueError("reward.nla_weight must be non-negative.")

        if self.rl_train.epochs <= 0:
            raise ValueError("rl_train.epochs must be positive.")
        if self.rl_train.trajectory_length <= 0:
            raise ValueError("rl_train.trajectory_length must be positive.")
        if not 0 <= self.rl_train.discount_factor <= 1:
            raise ValueError("rl_train.discount_factor must be in [0, 1].")
        if self.rl_train.critic_num_bins < 2:
            raise ValueError("rl_train.critic_num_bins must be at least two.")
        if not 0 < self.rl_train.initial_state_randomization_rate < 1:
            raise ValueError("rl_train.initial_state_randomization_rate must be in (0, 1).")
        if self.rl_train.actor_optimizer_name.lower() != "adamw":
            raise ValueError("The RL actor optimizer must be AdamW.")
        if self.rl_train.actor_lr <= 0:
            raise ValueError("rl_train.actor_lr must be positive.")
        if self.rl_train.actor_weight_decay < 0:
            raise ValueError("rl_train.actor_weight_decay must be non-negative.")
        beta1, beta2 = self.rl_train.actor_adamw_betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("rl_train.actor_adamw_betas must be in [0, 1).")
        if self.rl_train.actor_adamw_eps <= 0:
            raise ValueError("rl_train.actor_adamw_eps must be positive.")
        if self.rl_train.critic_optimizer_name.lower() != "sgd":
            raise ValueError("The RL critic optimizer must be SGD.")
        if self.rl_train.critic_lr <= 0:
            raise ValueError("rl_train.critic_lr must be positive.")
        if not 0 <= self.rl_train.critic_momentum < 1:
            raise ValueError("rl_train.critic_momentum must be in [0, 1).")
        if self.rl_train.critic_weight_decay < 0:
            raise ValueError("rl_train.critic_weight_decay must be non-negative.")
        if self.rl_train.scheduler_name != "step_halfway":
            raise ValueError("The RL scheduler must be step_halfway.")
        if not 0 < self.rl_train.lr_decay_fraction < 1:
            raise ValueError("rl_train.lr_decay_fraction must be in (0, 1).")
        if not 0 < self.rl_train.lr_decay_factor < 1:
            raise ValueError("rl_train.lr_decay_factor must be in (0, 1).")
        if self.rl_train.policy_update_mode not in {"full", "subset"}:
            raise ValueError("rl_train.policy_update_mode must be 'full' or 'subset'.")
        if self.rl_train.policy_update_subset_size <= 0:
            raise ValueError("rl_train.policy_update_subset_size must be positive.")
        if self.rl_train.policy_update_batch_size <= 0:
            raise ValueError("rl_train.policy_update_batch_size must be positive.")
        if (
            self.rl_train.policy_update_mode == "subset"
            and self.rl_train.policy_update_batch_size > self.rl_train.policy_update_subset_size
        ):
            raise ValueError("RL policy update batch size cannot exceed subset size.")
        if self.rl_train.checkpoint_interval <= 0:
            raise ValueError("rl_train.checkpoint_interval must be positive.")
        checkpoint_filename = self.rl_train.checkpoint_filename
        if not checkpoint_filename or Path(checkpoint_filename).name != checkpoint_filename:
            raise ValueError("rl_train.checkpoint_filename must be a plain filename.")
        resume_path = self.rl_train.resume_checkpoint_path
        if resume_path is not None and not resume_path.is_absolute():
            raise ValueError("rl_train.resume_checkpoint_path must be None or absolute.")
        if not isinstance(self.rl_train.overwrite, bool):
            raise TypeError("rl_train.overwrite must be a boolean.")
        if not self.rl_train.output_dir.is_absolute():
            raise ValueError("rl_train.output_dir must be absolute.")

        if self.cleaning.trajectory_length <= 0:
            raise ValueError("cleaning.trajectory_length must be positive.")
        if not self.cleaning.checkpoint_path.is_absolute():
            raise ValueError("cleaning.checkpoint_path must be absolute.")
        if not self.cleaning.output_dir.is_absolute():
            raise ValueError("cleaning.output_dir must be absolute.")
        cleaning_filenames = (self.cleaning.artifact_filename, self.cleaning.corrected_labels_filename)
        if any(not filename or Path(filename).name != filename for filename in cleaning_filenames):
            raise ValueError("Cleaning outputs must use plain filenames.")
        if self.cleaning.artifact_filename == self.cleaning.corrected_labels_filename:
            raise ValueError("Cleaning output filenames must be different.")
        if not isinstance(self.cleaning.overwrite, bool):
            raise TypeError("cleaning.overwrite must be a boolean.")

        supported_metrics = {"macro_f1", "balanced_accuracy", "loss"}
        checkpoint_metrics = [item.metric for item in self.checkpoint.best]
        if set(checkpoint_metrics) != supported_metrics or len(checkpoint_metrics) != 3:
            raise ValueError("checkpoint.best must contain macro_f1, balanced_accuracy, and loss once each.")
        if len({item.filename for item in self.checkpoint.best}) != 3:
            raise ValueError("Best checkpoint filenames must be unique.")


CFG = Config()
