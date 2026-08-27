"""CIFAR-10 RLNLC experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path("/root/project/rlnlc")


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: Path = PROJECT_ROOT / "cifar10"
    download: bool = True

    classes: tuple[int, ...] = tuple(range(10))
    train_samples: int = 50_000  # 10_000, 20_000, 50_000; 10의 배수
    subset_seed: int = 0  # train_samples < 50_000일 때 균등 표본 추출 시드
    noise_type: str = "idn"  # "symmetric" 또는 "idn"
    noise_rate: float = 0.50  # 목표 노이즈 비율: 0.2, 0.4, 0.5
    idn_flip_rate_std: float = 0.10  # Xia et al. IDN truncated-normal 표준편차

    seed: int = 0  # 노이즈 생성 및 전체 실험 재현 시드
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "cifar_resnet34"  # "cifar_resnet18" 또는 "cifar_resnet34"
    pretrained: bool = False


@dataclass(frozen=True, slots=True)
class TrainingAugmentationConfig:
    enabled: bool = True
    random_crop_padding: int = 4  # CIFAR 표준값; 0이면 crop 증강 제거
    horizontal_flip_probability: float = 0.5  # 0.0~1.0


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    model_id: str = "exp12_warmup"
    epochs: int = 50  # warm-up 학습 길이
    batch_size: int = 1_024  # 메모리에 맞춰 128, 512, 1_024
    eval_batch_size: int = 1_024  # 성능에는 영향 없음; 메모리·속도 조절
    optimizer: str = "sgd"
    learning_rate: float = 1e-2  # SGD 후보: 1e-3, 3e-3, 1e-2
    momentum: float = 0.9  # SGD momentum: 보통 0.9
    weight_decay: float = 5e-4  # L2 규제: 0, 1e-4, 5e-4
    lr_decay_fraction: float = 0.5  # 전체 epoch 중 LR 감소 시점 비율
    lr_decay_factor: float = 0.1  # 감소 시 LR에 곱할 값
    min_noisy_validation_accuracy: float = 0.0  # checkpoint 허용 하한; 0이면 비활성


@dataclass(frozen=True, slots=True)
class KNNConfig:
    k: int = 10  # 이웃 수: 5, 10, 20
    temperature: float = 0.5  # 거리 가중치 온도; 작을수록 가까운 이웃 강조
    query_chunk_size: int = 8_192  # 결과에는 영향 없음; KNN 검색 메모리·속도 조절
    reference_chunk_size: int = 65_536  # 결과에는 영향 없음; KNN 기준 청크 크기
    correction_chunk_size: int = 50_000  # 결과에는 영향 없음; correction 메모리·속도 조절


@dataclass(frozen=True, slots=True)
class RLConfig:
    epochs: int = 500  # 빠른 검증 100, 중간 200, 최종 재현 500
    trajectory_length: int = 10  # 한 epoch의 label-correction step 수
    initial_state_randomization_rate: float = 0.10  # 매 trajectory 초기 라벨 교란 비율
    feature_batch_size: int = 8_192  # 결과에는 영향 없음; feature 추출 메모리·속도 조절
    actor_batch_size: int = 128  # step당 Actor optimizer 미니배치: 128, 256, 512
    use_remaining_horizon: bool = True
    use_terminal_critic_update: bool = True

    actor_optimizer: str = "sgd"
    actor_learning_rate: float = 1e-2  # SGD 후보: 1e-3, 3e-3, 1e-2
    actor_momentum: float = 0.9  # SGD momentum: 보통 0.9
    actor_weight_decay: float = 5e-4  # L2 규제: 0, 1e-4, 5e-4

    critic_optimizer: str = "sgd"  # "sgd" 또는 "adam"
    critic_learning_rate: float = 1e-2  # SGD일 때 사용: 1e-3, 3e-3, 1e-2
    critic_adam_learning_rate: float = 1e-3  # Adam일 때 사용: 1e-4, 3e-4, 1e-3
    critic_momentum: float = 0.9  # SGD에서만 사용
    critic_weight_decay: float = 5e-4  # SGD에서만 사용; Adam 선택 시 자동으로 0
    critic_num_bins: int = 100  # label-consistency histogram bin 수
    critic_hidden_dims: tuple[int, ...] = (128, 64)  # 예: (128, 64), (256, 128, 64)

    discount_factor: float = 0.9  # TD discount gamma: 0.0~1.0
    reward_nla_weight: float = 0.5  # log_reward에서 NLA 항의 가중치
    lr_decay_epoch: int = 250  # 총 epoch와 무관한 SGD Actor/Critic LR 감소 epoch
    lr_decay_factor: float = 0.1  # SGD LR 감소 배율; Adam Critic에는 미적용

    @property
    def effective_critic_options(self) -> tuple[float, float, float, bool]:
        if self.critic_optimizer == "adam":
            return self.critic_adam_learning_rate, 0.0, 0.0, False
        return self.critic_learning_rate, self.critic_momentum, self.critic_weight_decay, True


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    trajectory_length: int = 25  # 최종 Actor로 반복 correction할 step 수


@dataclass(frozen=True, slots=True)
class KNNQualityConfig:
    visualization_samples: int = 10_000  # 시각화 표본 수; 클래스 수로 나누어져야 함
    pca_dimensions: int = 50  # UMAP 전 PCA 차원
    umap_neighbors: int = 15  # 지역 구조 범위: 5, 15, 30
    umap_min_dist: float = 0.1  # 군집 압축도: 0.0~1.0
    seed: int = 0  # 시각화 재현 시드


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    corrected_label_source: str = "rl"
    initialization: str = "last_actor"
    evaluation_checkpoint: str = "accuracy"
    epochs: int = 100  # corrected label fine-tuning 길이
    batch_size: int = 1_024  # 메모리에 맞춰 128, 512, 1_024
    optimizer: str = "sgd"
    learning_rate: float = 1e-2  # SGD 후보: 1e-3, 3e-3, 1e-2
    momentum: float = 0.9  # SGD momentum: 보통 0.9
    weight_decay: float = 5e-4  # L2 규제: 0, 1e-4, 5e-4
    lr_decay_fraction: float = 0.5  # 전체 epoch 중 LR 감소 시점 비율
    lr_decay_factor: float = 0.1  # 감소 시 LR에 곱할 값


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: Path = PROJECT_ROOT / "cifar_output"
    experiment_name: str = "exp12"
    warmup_experiment_name: str = "exp12"
    warmup_checkpoint_name: str = "warmup.pt"
    actor_best_checkpoint_name: str = "actor_best.pt"
    actor_last_checkpoint_name: str = "actor_last.pt"
    critic_best_checkpoint_name: str = "critic_best.pt"
    critic_last_checkpoint_name: str = "critic_last.pt"
    corrected_labels_name: str = "train_corrected_soft_labels.npy"
    knn_corrected_labels_name: str = "train_knn_corrected_soft_labels.npy"
    finetune_best_accuracy_checkpoint_name: str = "finetune_best_accuracy.pt"
    finetune_best_loss_checkpoint_name: str = "finetune_best_loss.pt"
    finetune_last_checkpoint_name: str = "finetune_last.pt"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_channels_last: bool = True
    cudnn_benchmark: bool = True
    evaluate_batch_size: int = 1_024  # 성능에는 영향 없음; 평가 메모리·속도 조절
    overwrite_noise: bool = False
    overwrite_warmup: bool = False
    overwrite_rl: bool = False
    overwrite_correction: bool = False
    overwrite_knn_correction: bool = False
    overwrite_finetune: bool = False
    overwrite_evaluate: bool = False
    overwrite_knn_quality: bool = False


@dataclass(frozen=True, slots=True)
class CIFARConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: TrainingAugmentationConfig = field(default_factory=TrainingAugmentationConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    knn_quality: KNNQualityConfig = field(default_factory=KNNQualityConfig)
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
        noise_name = f"noise{self.noise_tag}"
        if self.data.noise_type == "idn":
            std_tag = f"{self.data.idn_flip_rate_std:g}".replace(".", "p")
            noise_name = f"noise_idn{self.noise_tag}_std{std_tag}"
        name = (
            f"cifar10_train{self.data.train_samples}_subsetseed{self.data.subset_seed}_"
            f"{noise_name}_seed{self.data.seed}"
        )
        return self.data_root / name

    @property
    def experiment_output_dir(self) -> Path:
        return self.output_root / self.output.experiment_name

    @property
    def warmup_output_dir(self) -> Path:
        return self.output_root / self.output.warmup_experiment_name / "warmup"

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
    def knn_quality_visualization_samples(self) -> int:
        return min(self.knn_quality.visualization_samples, self.data.train_samples)

    @property
    def log_output_dir(self) -> Path:
        return self.experiment_output_dir / "logs"

    @property
    def rl_output_dir(self) -> Path:
        return self.log_output_dir / "rl"

    @property
    def rl_model_dir(self) -> Path:
        return self.experiment_output_dir / "model_rl"

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
    def knn_output_dir(self) -> Path:
        return self.experiment_output_dir / "knn"

    @property
    def knn_correction_output_dir(self) -> Path:
        return self.knn_output_dir / "logs" / "correction"

    @property
    def knn_corrected_labels_path(self) -> Path:
        return self.knn_output_dir / "model" / self.output.knn_corrected_labels_name

    @property
    def finetune_corrected_labels_path(self) -> Path:
        return {
            "rl": self.corrected_labels_path,
            "knn": self.knn_corrected_labels_path,
        }[self.finetune.corrected_label_source]

    @property
    def finetune_source_output_dir(self) -> Path:
        return self.experiment_output_dir if self.finetune.corrected_label_source == "rl" else self.knn_output_dir

    @property
    def finetune_initial_checkpoint_path(self) -> Path:
        return {
            "warmup": self.warmup_checkpoint_path,
            "best_actor": self.actor_best_checkpoint_path,
            "last_actor": self.actor_last_checkpoint_path,
        }[self.finetune.initialization]

    @property
    def finetune_output_dir(self) -> Path:
        return self.finetune_source_output_dir / "logs" / "finetune"

    @property
    def finetune_model_dir(self) -> Path:
        return self.finetune_source_output_dir / "model_finetune"

    @property
    def evaluate_output_dir(self) -> Path:
        return self.finetune_source_output_dir / "logs" / "evaluate"

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
        if self.model.name not in {"cifar_resnet18", "cifar_resnet34"}:
            raise ValueError("model.name must be 'cifar_resnet18' or 'cifar_resnet34'.")
        if self.model.pretrained:
            raise ValueError("The paper-style CIFAR ResNet must not be pretrained.")
        sgd_optimizers = (self.warmup.optimizer, self.rl.actor_optimizer, self.finetune.optimizer)
        if any(name.lower() != "sgd" for name in sgd_optimizers):
            raise ValueError("CIFAR warm-up, actor, and fine-tuning optimizers must be SGD.")
        if self.rl.critic_optimizer not in {"sgd", "adam"}:
            raise ValueError("critic_optimizer must be 'sgd' or 'adam'.")
        if self.data.classes != tuple(range(10)):
            raise ValueError("CIFAR-10 classes must be 0 through 9.")
        if not 0 < self.data.train_samples <= 50_000 or self.data.train_samples % 10:
            raise ValueError("train_samples must be in [1, 50000] and divisible by 10.")
        if not 0 <= self.data.noise_rate < 1:
            raise ValueError("noise_rate must be in [0, 1).")
        if self.data.noise_type not in {"symmetric", "idn"}:
            raise ValueError("noise_type must be 'symmetric' or 'idn'.")
        if self.data.idn_flip_rate_std <= 0:
            raise ValueError("idn_flip_rate_std must be positive.")
        if not 0 < self.rl.actor_batch_size <= self.data.train_samples:
            raise ValueError("actor_batch_size must be in [1, train_samples].")
        if self.finetune.initialization not in {"warmup", "best_actor", "last_actor"}:
            raise ValueError("Invalid fine-tuning initialization.")
        if self.finetune.corrected_label_source not in {"rl", "knn"}:
            raise ValueError("corrected_label_source must be 'rl' or 'knn'.")
        if self.finetune.evaluation_checkpoint not in {"accuracy", "loss"}:
            raise ValueError("evaluation_checkpoint must be accuracy or loss.")
        if self.knn.k >= self.data.train_samples:
            raise ValueError("k must be smaller than train_samples.")
        if not 0 < self.rl.initial_state_randomization_rate < 1:
            raise ValueError("initial_state_randomization_rate must be in (0, 1).")
        if not 0 <= self.rl.discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1].")
        if not 1 <= self.rl.lr_decay_epoch <= self.rl.epochs:
            raise ValueError("lr_decay_epoch must be in [1, rl.epochs].")
        if not 0 < self.rl.lr_decay_factor <= 1:
            raise ValueError("rl.lr_decay_factor must be in (0, 1].")
        if not self.rl.critic_hidden_dims or any(width <= 0 for width in self.rl.critic_hidden_dims):
            raise ValueError("critic_hidden_dims must contain positive widths.")
        if self.runtime.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16.")

        positive = (
            self.warmup.epochs,
            self.warmup.batch_size,
            self.warmup.eval_batch_size,
            self.rl.epochs,
            self.rl.trajectory_length,
            self.rl.feature_batch_size,
            self.rl.actor_batch_size,
            self.correction.trajectory_length,
            self.knn_quality.visualization_samples,
            self.knn_quality.pca_dimensions,
            self.knn_quality.umap_neighbors,
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
            self.rl.critic_adam_learning_rate,
            self.finetune.learning_rate,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Epochs, batch sizes, chunk sizes, and learning rates must be positive.")
        if self.knn_quality_visualization_samples % len(self.data.classes):
            raise ValueError("KNN visualization samples must be divisible by the number of classes.")
        if self.knn_quality.umap_neighbors >= self.knn_quality_visualization_samples:
            raise ValueError("UMAP neighbors must be smaller than visualization samples.")
        if not 0 <= self.knn_quality.umap_min_dist <= 1:
            raise ValueError("UMAP min_dist must be in [0, 1].")


CONFIG = CIFARConfig()
CONFIG.validate()
