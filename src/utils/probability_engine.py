# from datetime import datetime, timedelta
# import numpy as np
# import logging

# class ProbabilityEngine:
#     def __init__(self, db):
#         self.db = db
#         self.active_bots = {}
#         self.weight_cache = {}
#         self.last_weight_update = None
        
#     def update_rankings(self):
#         """Update bot rankings based on current metrics"""
#         try:
#             # Update weights if needed
#             if self._should_update_weights():
#                 self._update_optimal_weights()

#             # Get current bot metrics
#             bot_metrics = self._get_current_metrics()

#             # Calculate new rankings
#             rankings = self._calculate_rankings(bot_metrics)

#             # Update database with new rankings
#             self._update_bot_rankings(rankings)

#             # Manage bot activation based on new rankings
#             self._manage_bot_activation(rankings)

#         except Exception as e:
#             logging.error(f"Ranking update failed: {str(e)}")

#     def _should_update_weights(self):
#         """Check if weights need updating"""
#         if not self.last_weight_update:
#             return True
#         return datetime.now() - self.last_weight_update > timedelta(hours=1)

#     def _update_optimal_weights(self):
#         """Update optimal weights based on system performance"""
#         timeframes = ['1h', '4h', '1d', '1w', '1m']
#         metrics = ['win_rate', 'profit_factor', 'sharpe_ratio']

#         # Generate weight combinations
#         weights = {}
#         for tf in timeframes:
#             for metric in metrics:
#                 weights[f"{metric}_{tf}"] = np.random.uniform(0, 1)

#         # Normalize weights to sum to 1
#         total = sum(weights.values())
#         for key in weights:
#             weights[key] /= total

#         self.weight_cache = weights
#         self.last_weight_update = datetime.now()

#     def _calculate_rankings(self, metrics):
#         """Calculate bot rankings using current weights"""
#         rankings = []
        
#         for bot_id, bot_metrics in metrics.items():
#             score = 0
#             for metric, weight in self.weight_cache.items():
#                 if metric in bot_metrics:
#                     score += bot_metrics[metric] * weight
#             rankings.append((bot_id, score))
            
#         # Sort by score descending
#         rankings.sort(key=lambda x: x[1], reverse=True)
#         return rankings
        
#     def _manage_bot_activation(self, rankings):
#         """Activate/deactivate bots based on rankings"""
#         # Get available funds
#         available_funds = self._get_available_funds()
#         allocated_funds = 0
        
#         for bot_id, score in rankings:
#             bot = self.active_bots.get(bot_id)
#             if not bot:
#                 continue
                
#             # Check if we can afford to activate this bot
#             required_funds = bot.get_required_funds()
#             if allocated_funds + required_funds <= available_funds:
#                 bot.activate()
#                 allocated_funds += required_funds
#             else:
#                 bot.deactivate()
