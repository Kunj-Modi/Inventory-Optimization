"""

Inventory Management Environment with CRITICAL FIXES:
1. Unified shared reward signal for both agents
2. Simplified cost function without ad-hoc penalties
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional, List
import configs.config as config

class InventoryMAMDPEnv(gym.Env):
    """
    VERSION: Environment with unified shared reward and simplified cost function
    """

    def __init__(self, config_obj=None):
        super(InventoryMAMDPEnv, self).__init__()

        # Use provided config or default to global config
        self.cfg = config_obj if config_obj else config.config.env

        # Store core parameters
        self.num_products = self.cfg.NUM_PRODUCTS
        self.fg_lead_time = self.cfg.FG_LEAD_TIME
        self.rm_lead_time = self.cfg.RM_LEAD_TIME

        # Cost parameters
        self.cost_fg_holding = np.array(self.cfg.COST_FG_HOLDING)
        self.cost_fg_shortage = np.array(self.cfg.COST_FG_SHORTAGE)
        self.cost_rm_holding = self.cfg.COST_RM_HOLDING
        self.cost_rm_order = self.cfg.COST_RM_ORDER

        # Demand simulation parameters
        self.demand_mean = np.array(self.cfg.DEMAND_MEAN)
        self.demand_std = np.array(self.cfg.DEMAND_STD)
        self.seasonality_amplitude = self.cfg.SEASONALITY_AMPLITUDE

        # Define action spaces for both agents
        self.action_space_fg = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_products,),
            dtype=np.float32
        )
        self.action_space_rm = spaces.Box(
            low=0,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )

        # Define observation spaces - UPDATED for FG agent
        self.observation_space = spaces.Dict({
            'fg_agent': self._get_fg_observation_space(),
            'rm_agent': self._get_rm_observation_space()
        })

        # Initial conditions
        self.current_step = 0
        self.fg_inventory = np.array(self.cfg.INITIAL_INVENTORY_FG, dtype=np.float32)
        self.rm_inventory = self.cfg.INITIAL_INVENTORY_RM
        self.fg_pipeline = np.zeros((self.num_products, self.fg_lead_time), dtype=np.float32)
        self.rm_pipeline = np.full(self.rm_lead_time, 15.0, dtype=np.float32)
        self.demand_history = []
        self.max_steps = 365
        self.rm_stockout_count = 0

        if config.config.colab.DEBUG_MODE:
            print("Inventory Environment initialized successfully")
            print(f"Products: {self.num_products}, FG Lead Time: {self.fg_lead_time}, RM Lead Time: {self.rm_lead_time}")

    def _get_fg_observation_space(self) -> spaces.Box:
        """Enhanced FG observation space"""
        # New calculation: 5 + 15 + 4 + 3 + 4 + 2 + 4 = 37 dimensions
        obs_dim = (self.num_products +                    # current FG inventory (5)
                  self.num_products * self.fg_lead_time + # FG pipeline (15)
                  4 +                                    # demand metrics (4)
                  3 +                                    # historical patterns (3)
                  4 +                                    # production metrics (4)
                  2 +                                    # temporal features (2)
                  4)                                     # RM info (4)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_rm_observation_space(self) -> spaces.Box:
        # New dimension: 1 + 5 + 4 + 4 + 3 + 3 = 20
        obs_dim = (1 +                    # current RM inventory
                  self.rm_lead_time +     # RM pipeline (5)
                  4 +                     # production metrics
                  4 +                     # demand metrics
                  3 +                     # historical patterns
                  3)                      # system state
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment with initial conditions"""
        self.fg_inventory = np.array(self.cfg.INITIAL_INVENTORY_FG, dtype=np.float32)
        self.rm_inventory = self.cfg.INITIAL_INVENTORY_RM
        self.fg_pipeline = np.zeros((self.num_products, self.fg_lead_time), dtype=np.float32)
        self.rm_pipeline = np.full(self.rm_lead_time, 15.0, dtype=np.float32)
        self.current_step = 0
        self.rm_stockout_count = 0

        # Initialize demand history for forecasting
        self.demand_history = []
        for _ in range(self.cfg.FORECAST_HORIZON):
            self.demand_history.append(self._simulate_demand())

        return self._get_obs()

    def _get_fg_observation(self) -> np.ndarray:
        """Enhanced FG observation with predictive features"""
        obs_components = []

        # 1. Current FG inventory (M values) - EXISTING
        obs_components.append(self.fg_inventory)

        # 2. FG pipeline flattened (M * FG_LEAD_TIME values) - EXISTING
        obs_components.append(self.fg_pipeline.flatten())

        # 3. Enhanced demand forecasting (NEW - like RM agent)
        forecast = self._generate_demand_forecast()
        if forecast.size > 0:
            total_forecast = np.sum(forecast, axis=0)
            demand_metrics = [
                np.max(total_forecast),           # Peak demand forecast
                np.mean(total_forecast),          # Average demand
                np.std(total_forecast),           # Demand variability
                total_forecast[0]                 # Immediate demand
            ]
        else:
            demand_metrics = [0, 0, 0, 0]
        obs_components.append(demand_metrics)

        # 4. Historical patterns (NEW - like RM agent)
        if len(self.demand_history) > 7:
            recent_demand = np.array(self.demand_history[-7:])
            total_recent = np.sum(recent_demand, axis=1)
            historical_metrics = [
                np.max(total_recent),
                np.mean(total_recent),
                np.std(total_recent)
            ]
        else:
            historical_metrics = [0, 0, 0]
        obs_components.append(historical_metrics)

        # 5. Production efficiency metrics (NEW)
        production_metrics = [
            np.sum(self.fg_pipeline),             # Total planned production
            np.max(self.fg_pipeline),             # Peak production
            np.std(self.fg_pipeline),             # Production variability
            self.rm_inventory / (np.sum(self.fg_pipeline) + 1e-6)  # RM coverage ratio
        ]
        obs_components.append(production_metrics)

        # 6. Temporal features (EXISTING)
        day_of_year = self.current_step % 365
        temporal_features = [
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365)
        ]
        obs_components.append(temporal_features)

        # 7. RM inventory information (EXISTING but enhanced)
        rm_info = [
            self.rm_inventory,                    # Current RM Inventory level
            np.sum(self.rm_pipeline),             # Total RM in pipeline
            self.rm_stockout_count,               # Historical reliability
            np.mean(self.rm_pipeline)             # Average pipeline flow
        ]
        obs_components.append(rm_info)

        return np.concatenate([arr.flatten() if hasattr(arr, 'flatten') else np.array(arr)
                              for arr in obs_components], dtype=np.float32)

    def _get_rm_observation(self) -> np.ndarray:
        """Enhanced RM observation with predictive features"""
        obs_components = []

        # 1. Current RM status (existing)
        obs_components.append([self.rm_inventory])
        obs_components.append(self.rm_pipeline)

        # 2. Enhanced production signals (NEW)
        production_metrics = [
            np.sum(self.fg_pipeline[:, -1]),  # Immediate production needs
            np.sum(self.fg_pipeline),         # Total pipeline production
            np.max(self.fg_pipeline),         # Peak production demand
            np.std(self.fg_pipeline)          # Production variability
        ]
        obs_components.append(production_metrics)

        # 3. Enhanced demand forecasting (NEW)
        forecast = self._generate_demand_forecast()
        if forecast.size > 0:
            total_forecast = np.sum(forecast, axis=0)
            demand_metrics = [
                np.max(total_forecast),       # Peak demand forecast
                np.mean(total_forecast),      # Average demand
                np.std(total_forecast),       # Demand variability
                total_forecast[0]             # Immediate demand
            ]
        else:
            demand_metrics = [0, 0, 0, 0]
        obs_components.append(demand_metrics)

        # 4. Historical patterns (NEW)
        if len(self.demand_history) > 7:
            recent_demand = np.array(self.demand_history[-7:])
            total_recent = np.sum(recent_demand, axis=1)
            historical_metrics = [
                np.max(total_recent),
                np.mean(total_recent),
                np.std(total_recent)
            ]
        else:
            historical_metrics = [0, 0, 0]
        obs_components.append(historical_metrics)

        # 5. System state awareness (NEW)
        system_state = [
            self.current_step / 365.0,        # Training progress
            min(self.rm_stockout_count, 10),  # Historical failures (capped)
            np.sum(self.rm_pipeline > 0) / len(self.rm_pipeline)  # Pipeline utilization
        ]
        obs_components.append(system_state)

        return np.concatenate([arr.flatten() if hasattr(arr, 'flatten') else np.array(arr)
                              for arr in obs_components], dtype=np.float32)

    def step(self, joint_action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute one timestep with CRITICAL FIX: unified shared reward"""

        # Extract actions
        fg_action = joint_action['fg_agent']
        rm_action = joint_action['rm_agent'][0]

        # Clip actions to be non-negative
        fg_action = np.clip(fg_action, 0, None)
        rm_action = max(0, rm_action)

        # Action clipping for FG agent
        max_reasonable_production = self.rm_inventory * 1.5
        current_total_planned = np.sum(fg_action)

        if current_total_planned > max_reasonable_production:
            scale_factor = max_reasonable_production / current_total_planned
            fg_action = fg_action * scale_factor

        # Cap individual product orders
        max_per_product = 25.0
        fg_action = np.clip(fg_action, 0, max_per_product)

        # 1. Fulfill demand and update FG inventory
        realized_demand = self._simulate_demand()
        lost_sales, fulfilled_demand = self._fulfill_demand(realized_demand)

        # 2. Update FG production pipeline (FIFO)
        self._update_fg_pipeline(fg_action)

        # 3. Update RM inventory and handle production constraints
        actual_production = self._update_rm_inventory(fg_action)

        # 4. Update RM procurement pipeline (FIFO)
        self._update_rm_pipeline(rm_action)

        # 6. Unified shared reward for both agents
        (shared_reward, fg_holding_cost, fg_shortage_cost, rm_holding_cost, rm_order_cost, rm_stockout_penalty) = self._calculate_total_reward_simplified(fulfilled_demand, lost_sales, rm_action)

        # 5. total system cost - SIMPLIFIED VERSION
        total_cost = fg_holding_cost + rm_holding_cost + rm_order_cost

        # 7. Update demand history for forecasting
        self.demand_history.append(realized_demand)
        if len(self.demand_history) > self.cfg.FORECAST_HORIZON:
            self.demand_history.pop(0)

        # 8. Increment step and check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # 9. Get new observations
        obs = self._get_obs()

        # 10. Prepare info dictionary
        info = {
            'step': self.current_step,
            'total_cost': total_cost,
            'fg_inventory': self.fg_inventory.copy(),
            'rm_inventory': self.rm_inventory,
            'lost_sales': lost_sales,
            'fulfilled_demand': fulfilled_demand,
            'rm_stockout_count': self.rm_stockout_count,
            'actual_production': actual_production,
            'planned_production': fg_action,
            # ADD cost components for detailed analysis
            'fg_holding_cost': fg_holding_cost,
            'fg_shortage_cost': fg_shortage_cost,
            'rm_holding_cost': rm_holding_cost,
            'rm_stockout_penalty': rm_stockout_penalty,
            'rm_order_cost': rm_order_cost
        }

        # CRITICAL FIX: Return same reward for both agents
        rewards = {
            'fg_agent': shared_reward,
            'rm_agent': shared_reward
        }

        return obs, rewards, done, info

    def _calculate_total_reward_simplified(self, fulfilled_demand: np.ndarray, lost_sales: np.ndarray, rm_action: float) -> float:
        """Enhanced cost function with CRITICAL RM stockout penalty"""

        # FG costs
        fg_holding_cost = np.sum(self.cost_fg_holding * self.fg_inventory)
        fg_shortage_cost = np.sum(self.cost_fg_shortage * lost_sales)

        # RM costs - CRITICAL: Massive penalty for stockouts
        rm_holding_cost = self.cost_rm_holding * self.rm_inventory
        rm_order_cost = self.cost_rm_order if rm_action > 0 else 0

        # FIXED: Smooth quadratic penalty for RM stockouts
        rm_stockout_penalty = 0
        planned_production = np.sum(self.fg_pipeline[:, -1]) # New production planned
        if planned_production > 0:
            # Ultra-smooth penalty with linear-quadratic combination
            rm_shortage = max(0, planned_production - self.rm_inventory)
            if rm_shortage > 0:
                # Combined linear + quadratic for very smooth gradients
                rm_stockout_penalty = 1.0 * rm_shortage + 0.5 * (rm_shortage ** 2)
            else:
                rm_stockout_penalty = 0

        total_reward = -(fg_holding_cost + fg_shortage_cost +
                      rm_holding_cost + rm_order_cost + rm_stockout_penalty)

        return (total_reward, fg_holding_cost, fg_shortage_cost, rm_holding_cost, rm_order_cost, rm_stockout_penalty)

    def _simulate_demand(self) -> np.ndarray:
        """Simulate daily demand for each product"""
        base_demand = np.random.normal(self.demand_mean, self.demand_std)
        base_demand = np.clip(base_demand, 0, self.cfg.MAX_DAILY_DEMAND)

        day_of_week = self.current_step % 7
        seasonality_factor = 1.0 + self.seasonality_amplitude * np.sin(2 * np.pi * day_of_week / 7)

        demand = base_demand * seasonality_factor
        return np.clip(demand, 0, None).astype(np.float32)

    def _fulfill_demand(self, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fulfill customer demand and calculate lost sales"""
        fulfilled = np.minimum(self.fg_inventory, demand)
        lost_sales = demand - fulfilled
        self.fg_inventory -= fulfilled
        self.fg_inventory = np.clip(self.fg_inventory, 0, None)
        return lost_sales, fulfilled

    def _update_fg_pipeline(self, new_production: np.ndarray):
        """Update finished goods pipeline with FIFO logic"""
        for d in range(self.fg_lead_time - 1):
            self.fg_pipeline[:, d] = self.fg_pipeline[:, d + 1]
        self.fg_pipeline[:, -1] = new_production
        completed_production = self.fg_pipeline[:, 0].copy()
        self.fg_inventory += completed_production
        self.fg_pipeline[:, 0] = 0

    def _update_rm_inventory(self, planned_production: np.ndarray) -> np.ndarray:
        """Update raw material inventory and handle production constraints"""
        rm_needed = np.sum(planned_production)

        if self.rm_inventory >= rm_needed:
            self.rm_inventory -= rm_needed
            actual_production = planned_production
        else:
            scale_factor = self.rm_inventory / rm_needed if rm_needed > 0 else 0
            actual_production = planned_production * scale_factor
            self.rm_inventory = 0
            self.rm_stockout_count += 1

        return actual_production

    def _update_rm_pipeline(self, new_order: float):
        """Update raw material pipeline with FIFO logic"""
        for d in range(self.rm_lead_time - 1):
            self.rm_pipeline[d] = self.rm_pipeline[d + 1]
        self.rm_pipeline[-1] = new_order
        rm_arrival = self.rm_pipeline[0]
        self.rm_inventory += rm_arrival
        self.rm_pipeline[0] = 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Compile current system state into observations for both agents"""
        return {
            'fg_agent': self._get_fg_observation(),
            'rm_agent': self._get_rm_observation()
        }

    def _generate_demand_forecast(self) -> np.ndarray:
        """Generate demand forecast using simple moving average"""
        if not self.demand_history:
            return np.zeros((self.num_products, self.cfg.FORECAST_HORIZON))

        history_array = np.array(self.demand_history)
        forecast = []
        for product_idx in range(self.num_products):
            product_history = history_array[:, product_idx]
            product_forecast = np.full(self.cfg.FORECAST_HORIZON, product_history[-1])
            forecast.append(product_forecast)

        return np.array(forecast, dtype=np.float32)

    def render(self, mode='human'):
        """Render the current state of the environment"""
        print(f"\n== Step {self.current_step} ==")
        print(f"FG Inventory: {self.fg_inventory}")
        print(f"RM Inventory: {self.rm_inventory:.1f}")
        print(f"FG Pipeline shape: {self.fg_pipeline.shape}")
        print(f"RM Pipeline: {self.rm_pipeline}")

    def close(self):
        """Clean up environment resources"""
        pass
