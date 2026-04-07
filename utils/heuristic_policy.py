"""
Simplified heuristic policy for effective imitation learning
"""

import numpy as np
from typing import Dict
import configs.config as config

def calculate_base_stock_action(state: np.ndarray, agent_type: str, config_obj=None) -> np.ndarray:
    """
    SIMPLIFIED VERSION: Base-stock policy that is easier to imitate
    """
    cfg = config_obj if config_obj else config.config

    if agent_type == 'fg_agent':
        return _fg_base_stock(state, cfg)
    elif agent_type == 'rm_agent':
        return _rm_base_stock(state, cfg)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def _fg_base_stock(state: np.ndarray, cfg) -> np.ndarray:
    """Enhanced base-stock policy for Finished Goods agent with new state structure"""

    num_products = cfg.env.NUM_PRODUCTS
    fg_lead_time = cfg.env.FG_LEAD_TIME

    # Extract from enhanced state structure (37 dimensions)
    # Structure: 5(inv) + 15(pipe) + 4(demand) + 3(hist) + 4(prod) + 2(temp) + 4(RM)
    inventory_start = 0
    inventory_end = num_products  # 5
    pipeline_end = inventory_end + num_products * fg_lead_time  # 5 + 15 = 20
    demand_metrics_start = pipeline_end  # 20
    historical_start = demand_metrics_start + 4  # 24
    production_start = historical_start + 3  # 27
    temporal_start = production_start + 4  # 31
    rm_info_start = temporal_start + 2  # 33

    current_inventory = state[inventory_start:inventory_end]
    pipeline = state[inventory_end:pipeline_end].reshape(num_products, fg_lead_time)

    # Extract key metrics from new state structure
    immediate_demand = state[demand_metrics_start + 3]  # Last element of demand metrics
    demand_variability = state[demand_metrics_start + 2]  # Std of demand
    rm_inventory = state[rm_info_start]  # First element of RM info
    rm_pipeline_total = state[rm_info_start + 1]  # Second element of RM info

    # Dynamic base-stock based on demand and RM availability
    safety_factor = 1.0 + (demand_variability / 10.0)  # Adjust based on demand variability
    base_stock_levels = np.array([max(15, immediate_demand * 2 * safety_factor)] * num_products)

    # Adjust based on RM availability - more sophisticated
    total_rm_available = rm_inventory + rm_pipeline_total
    total_needed = np.sum(base_stock_levels)
    rm_scale = min(1.0, total_rm_available / (total_needed + 1e-6))
    base_stock_levels = base_stock_levels * rm_scale

    # Order up to base-stock level
    inventory_position = current_inventory + np.sum(pipeline, axis=1)
    order_quantities = np.maximum(0, base_stock_levels - inventory_position)

    # Reasonable caps based on RM availability
    max_order_per_product = min(25.0, total_rm_available / num_products)
    order_quantities = np.minimum(order_quantities, max_order_per_product)

    # Small noise for diversity
    noise = np.random.normal(0, 0.3, size=order_quantities.shape)
    order_quantities = np.maximum(0, order_quantities + noise)

    return order_quantities.astype(np.float32)

def _rm_base_stock(state: np.ndarray, cfg) -> np.ndarray:
    """Simple, smooth base-stock policy for Raw Material agent"""
    rm_lead_time = cfg.env.RM_LEAD_TIME
    forecast_horizon = cfg.env.FORECAST_HORIZON
    num_products = cfg.env.NUM_PRODUCTS

    # Extract state components
    current_inventory = state[0]
    pipeline = state[1:1 + rm_lead_time]
    production_signal = state[1 + rm_lead_time]
    aggregated_forecast = state[1 + rm_lead_time + 1:1 + rm_lead_time + 1 + forecast_horizon]

    # Calculate inventory position
    inventory_position = current_inventory + np.sum(pipeline)

    # Simple target calculation
    if len(aggregated_forecast) > 0:
        avg_daily_demand = np.mean(aggregated_forecast)
    else:
        avg_daily_demand = 8.0

    # Simple target: cover lead time demand + safety stock
    lead_time_demand = avg_daily_demand * num_products * rm_lead_time
    safety_stock = avg_daily_demand * num_products * 3  # 3 days of safety stock

    target_inventory = lead_time_demand + safety_stock

    # Order up to target
    order_quantity = max(0, target_inventory - inventory_position)

    # Small noise for diversity
    noise = np.random.normal(0, 1.0)
    order_quantity = max(0, order_quantity + noise)

    return np.array([order_quantity], dtype=np.float32)

def get_heuristic_action(state_dict: Dict[str, np.ndarray], config_obj=None) -> Dict[str, np.ndarray]:
    """Simplified heuristic policy for both agents"""
    cfg = config_obj if config_obj else config.config

    fg_action = _fg_base_stock(state_dict['fg_agent'], cfg)
    rm_action = _rm_base_stock(state_dict['rm_agent'], cfg)

    return {
        'fg_agent': fg_action,
        'rm_agent': rm_action
    }
