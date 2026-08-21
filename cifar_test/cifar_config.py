"""CIFAR-10 ResNet-18 experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path("/root/project/rlnlc")


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: Path = PROJECT_ROOT / "cifar10"
    download: bool = True

    classes: tuple[int, ...] = tuple(range(10))
    train_samples: int = 10_000
    subset_seed: int = 0
    noise_rate: float = 0.50

    seed: int = 0
    image_size: int = 32
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "cifar_resnet18"
    pretrained: bool = False


@dataclass(frozen=True, slots=True)
class TrainingAugmentationConfig:
    enabled: bool = True
    random_crop_padding: int = 4
    horizontal_flip_probability: float = 0.5


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    model_id: str = "exp1_warmup"
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
    update_mode: str = "full"
    subset_size: int = 5_000

    update_batch_size: int = 512
    record_change_diagnostics: bool = True
    change_diagnostic_probe_size: int = 50_000

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
class FineTuneConfig:
    initialization: str = "warmup"
    evaluation_checkpoint: str = "accuracy"
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_decay_fraction: float = 0.5
    lr_decay_factor: float = 0.1


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: Path = PROJECT_ROOT / "cifar_output"
    experiment_name: str = "exp1"
    warmup_checkpoint_name: str = "warmup.pt"
    actor_best_checkpoint_name: str = "actor_best.pt"
    actor_last_checkpoint_name: str = "actor_last.pt"
    critic_best_checkpoint_name: str = "critic_best.pt"
    critic_last_checkpoint_name: str = "critic_last.pt"
    corrected_labels_name: str = "train_corrected_soft_labels.npy"
    finetune_best_accuracy_checkpoint_name: str = "finetune_best_accuracy.pt"
    finetune_best_loss_checkpoint_name: str = "finetune_best_loss.pt"
    finetune_last_checkpoint_name: str = "finetune_last.pt"


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
    augmentation: TrainingAugmentationConfig = field(default_factory=TrainingAugmentationConfig)
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
        name = (
            f"cifar10_train{self.data.train_samples}_subsetseed{self.data.subset_seed}_"
            f"noise{self.noise_tag}_seed{self.data.seed}"
        )
        return self.data_root / name

    @property
    def experiment_output_dir(self) -> Path:
        return self.output_root / self.output.experiment_name

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
        return self.data.train_samples if self.rl.update_mode == "full" else self.rl.subset_size

    @property
    def actor_update_batch_size(self) -> int:
        return min(self.rl.update_batch_size, self.actor_update_samples)

    @property
    def change_diagnostic_probe_size(self) -> int:
        return min(self.rl.change_diagnostic_probe_size, self.data.train_samples)

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
        return {
            "warmup": self.warmup_checkpoint_path,
            "best_actor": self.actor_best_checkpoint_path,
            "last_actor": self.actor_last_checkpoint_path,
        }[self.finetune.initialization]

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
    def finetune_best_accuracy_checkpoint_path(self) -> Path:
        return self.finetune_model_dir / self.output.finetune_best_accuracy_checkpoint_name

    @property
    def finetune_best_loss_checkpoint_path(self) -> Path:
        return self.finetune_model_dir / self.output.finetune_best_loss_checkpoint_name

    @property
    def finetune_last_checkpoint_path(self) -> Path:
        return self.finetune_model_dir / self.output.finetune_last_checkpoint_name

    @property
    def finetune_evaluation_checkpoint_path(self) -> Path:
        return {
            "accuracy": self.finetune_best_accuracy_checkpoint_path,
            "loss": self.finetune_best_loss_checkpoint_path,
        }[self.finetune.evaluation_checkpoint]

    def validate(self) -> None:
        if not self.data_root.is_absolute() or not self.output_root.is_absolute():
            raise ValueError("Data and output roots must be absolute paths.")
        if self.model.pretrained:
            raise ValueError("The paper-style ResNet-18 must not be pretrained.")
        if self.data.classes != tuple(range(10)):
            raise ValueError("CIFAR-10 classes must be 0 through 9.")
        if not 0 < self.data.train_samples <= 50_000 or self.data.train_samples % 10:
            raise ValueError("train_samples must be in [1, 50000] and divisible by 10.")
        if not 0 <= self.data.noise_rate < 1:
            raise ValueError("noise_rate must be in [0, 1).")
        if self.rl.update_mode not in {"full", "subset"}:
            raise ValueError("update_mode must be 'full' or 'subset'.")
        if self.rl.update_mode == "subset" and not 0 < self.rl.subset_size <= self.data.train_samples:
            raise ValueError("subset_size must be in [1, train_samples].")
        if self.finetune.initialization not in {"warmup", "best_actor", "last_actor"}:
            raise ValueError("Invalid fine-tuning initialization.")
        if self.finetune.evaluation_checkpoint not in {"accuracy", "loss"}:
            raise ValueError("evaluation_checkpoint must be accuracy or loss.")
        if self.knn.k >= self.data.train_samples:
            raise ValueError("k must be smaller than train_samples.")
        if not 0 < self.rl.initial_state_randomization_rate < 1:
            raise ValueError("initial_state_randomization_rate must be in (0, 1).")
        if not 0 <= self.rl.discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1].")
        if self.runtime.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16.")

        positive = (
            self.warmup.epochs,
            self.warmup.batch_size,
            self.warmup.eval_batch_size,
            self.rl.epochs,
            self.rl.trajectory_length,
            self.rl.feature_batch_size,
            self.rl.update_batch_size,
            self.rl.change_diagnostic_probe_size,
            self.correction.trajectory_length,
            self.finetune.epochs,
            self.finetune.batch_size,
            self.runtime.evaluate_batch_size,
            self.knn.k,
            self.knn.temperature,
            self.knn.query_chunk_size,
            self.knn.reference_chunk_size,
            self.knn.correction_chunk_size,
            self.warmup.learning_rate,
            self.rl.actor_learning_rate,
            self.rl.critic_learning_rate,
            self.finetune.learning_rate,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Epochs, batch sizes, chunk sizes, and learning rates must be positive.")


CONFIG = ResNet18CIFARConfig()
CONFIG.validate()
