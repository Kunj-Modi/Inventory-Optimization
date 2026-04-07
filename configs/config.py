"""
Centralized configuration for the Multi-Agent DRL Inventory Management System
All hyperparameters and system constants are defined here.
"""

class EnvironmentConfig:
    """Configuration for the inventory management environment"""
    def __init__(self):
        # Core environment parameters
        self.NUM_PRODUCTS = 5    # Number of distinct finished goods (M)
        self.FG_LEAD_TIME = 3    # Production lead time for finished goods (days)
        self.RM_LEAD_TIME = 5    # Procurement lead time for raw materials (days)

        # Cost parameters
        self.COST_FG_HOLDING = [0.1] * 5    # Per-unit, per-day holding cost for each FG
        self.COST_FG_SHORTAGE = [10.0] * 5    # Per-unit shortage cost for each FG (lost sales)
        self.COST_RM_HOLDING = 0.02    # Per-unit, per-day holding cost for raw material
        self.COST_RM_ORDER = 0.5    # Fixed cost for placing raw material order

        # Demand and forecasting
        self.FORECAST_HORIZON = 7    # Days included in demand forecast
        self.INITIAL_INVENTORY_FG = [80] * 5    # Starting inventory for finished goods
        self.INITIAL_INVENTORY_RM = 1500    # Starting inventory for raw materials

        # Demand simulation parameters (for synthetic data generation)
        self.DEMAND_MEAN = [5.0] * 5    # Mean demand for each product
        self.DEMAND_STD = [2.0] * 5    # Standard deviation for demand
        self.SEASONALITY_AMPLITUDE = 2.0    # Amplitude of seasonal patterns
        self.MAX_DAILY_DEMAND = 15    # Maximum possible daily demand per product

class ModelConfig:
    """Configuration for neural network architectures"""
    def __init__(self):
        # CNN Time-Series Encoder
        self.HIDDEN_DIM = 32    # Neurons in hidden layers
        self.CNN_OUT_CHANNELS = 16    # Output channels for CNN encoder
        self.CNN_KERNEL_SIZE = 2    # Kernel size for causal convolutions
        self.CNN_NUM_LAYERS = 4    # Number of dilated convolutional layers

        # Transformer (for Phase 3)
        self.TRANSFORMER_D_MODEL = 128    # Hidden dimension for transformer
        self.TRANSFORMER_NHEAD = 8    # Number of attention heads
        self.TRANSFORMER_NUM_LAYERS = 4    # Number of transformer layers
        self.TRANSFORMER_DROPOUT = 0.1    # Dropout rate

        # Optimization
        self.LEARNING_RATE_ACTOR = 3e-4    # Learning rate for actor network
        self.LEARNING_RATE_CRITIC = 1e-3    # Learning rate for critic network
        self.OPTIMIZER_EPS = 1e-5    # Epsilon for Adam optimizer

class PPOConfig:
    """Configuration for PPO algorithm"""
    def __init__(self):
        # Advantage estimation
        self.GAMMA = 0.99    # Discount factor
        self.LAMBDA_GAE = 0.95    # GAE parameter

        # Clipping and optimization
        self.EPSILON_CLIP = 0.2    # PPO clipping parameter
        self.PPO_EPOCHS = 10    # Optimization epochs per update
        self.MINIBATCH_SIZE = 64    # Minibatch size
        self.ENTROPY_COEFF = 0.01    # Entropy bonus coefficient

        # Value function
        self.VALUE_COEFF = 0.5    # Value loss coefficient
        self.MAX_GRAD_NORM = 0.5    # Gradient clipping

class TrainingConfig:
    """Configuration for training process"""
    def __init__(self):
        # Main training
        self.TOTAL_TIMESTEPS = 200000    # Reduced for Colab testing
        self.UPDATE_TIMESTEPS = 2048    # Steps between PPO updates
        self.EVAL_FREQ = 10000    # Evaluation frequency
        self.SAVE_FREQ = 50000    # Model saving frequency

        # Imitation learning
        self.IMITATION_EPOCHS = 50    # Epochs for behavioral cloning
        self.IMITATION_BATCH_SIZE = 32    # Batch size for imitation learning
        self.EXPERT_DATASET_SIZE = 5000    # Size of expert dataset

        # Regularization
        self.LAMBDA_MONO = 0.1    # Monotonicity regularization weight - CRITICAL ADDITION

        # Base-stock levels for heuristic policy
        self.BASE_STOCK_FG = [20] * 5    # Base-stock levels for finished goods
        self.BASE_STOCK_RM = 30    # Base-stock level for raw material

class ColabConfig:
    """Colab-specific configuration"""
    def __init__(self):
        self.CHECKPOINT_DIR = "/content/checkpoints"
        self.TENSORBOARD_DIR = "/content/tensorboard"
        self.USE_GPU = True
        self.DEBUG_MODE = True

# Create global config object
class Config:
    def __init__(self):
        self.env = EnvironmentConfig()
        self.model = ModelConfig()
        self.ppo = PPOConfig()
        self.training = TrainingConfig()
        self.colab = ColabConfig()

    def print_config(self):
        """Print the entire configuration for verification"""
        print("== Configuration Summary ==")
        print(f"Environment: {self.env.NUM_PRODUCTS} products, FG lead time: {self.env.FG_LEAD_TIME}, RM lead time: {self.env.RM_LEAD_TIME}")
        print(f"Model: Hidden dim: {self.model.HIDDEN_DIM}, Learning rates: {self.model.LEARNING_RATE_ACTOR}/{self.model.LEARNING_RATE_CRITIC}")
        print(f"PPO: Gamma: {self.ppo.GAMMA}, Clip: {self.ppo.EPSILON_CLIP}, Epochs: {self.ppo.PPO_EPOCHS}")
        print(f"Training: Total steps: {self.training.TOTAL_TIMESTEPS}, Update freq: {self.training.UPDATE_TIMESTEPS}")
        print(f"Monotonicity Regularization: Lambda: {self.training.LAMBDA_MONO}")
        print(f"Colab: GPU: {self.colab.USE_GPU}, Debug: {self.colab.DEBUG_MODE}")

# Global config instance
config = Config()
