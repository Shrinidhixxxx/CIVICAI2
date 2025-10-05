"""
AutoML Optimizer for CivicMindAI using Optuna
Dynamically optimizes AI model parameters for better performance
"""

import optuna
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from collections import defaultdict
import statistics

class AutoMLOptimizer:
    """
    AutoML Optimizer for dynamic hyperparameter tuning and model optimization
    """
    
    def __init__(self, optimization_metric: str = "satisfaction_score"):
        self.optimization_metric = optimization_metric
        self.study = None
        self.optimization_history = []
        self.current_parameters = self._get_default_parameters()
        self.performance_metrics = defaultdict(list)
        
        # Optimization settings
        self.optimization_interval = 50  # Optimize every 50 queries
        self.query_count = 0
        self.last_optimization = None
        
        # Parameter bounds
        self.parameter_bounds = {
            "temperature": (0.1, 1.0),
            "max_tokens": (100, 500),
            "response_detail_level": (0.5, 2.0),
            "confidence_threshold": (0.5, 0.95),
            "context_window": (1, 5)
        }
        
        # Initialize Optuna study
        self._initialize_study()
        
        logging.info("AutoML Optimizer initialized")
    
    def _initialize_study(self):
        """
        Initialize Optuna study for hyperparameter optimization
        """
        try:
            # Create study with TPE sampler for efficient optimization
            self.study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )
            
            logging.info("Optuna study initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Optuna study: {e}")
            self.study = None
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for the AI model
        """
        return {
            "temperature": 0.7,
            "max_tokens": 200,
            "response_detail_level": 1.0,
            "confidence_threshold": 0.8,
            "context_window": 3,
            "optimization_version": 1
        }
    
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for optimization trial
        """
        try:
            suggested_params = {}
            
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                if param_name in ["max_tokens", "context_window"]:
                    # Integer parameters
                    suggested_params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
                else:
                    # Float parameters
                    suggested_params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            return suggested_params
            
        except Exception as e:
            logging.error(f"Error suggesting parameters: {e}")
            return self.current_parameters.copy()
    
    def evaluate_parameters(self, parameters: Dict[str, Any], 
                          performance_data: List[Dict[str, Any]]) -> float:
        """
        Evaluate parameter performance based on collected data
        """
        try:
            if not performance_data:
                return 0.5  # Neutral score for no data
            
            # Calculate multiple performance metrics
            satisfaction_scores = []
            response_time_scores = []
            accuracy_scores = []
            
            for data_point in performance_data:
                # Satisfaction score (0-1)
                satisfaction = data_point.get("user_satisfaction", 0.5)
                satisfaction_scores.append(satisfaction)
                
                # Response time score (inverse - faster is better)
                response_time = data_point.get("response_time", 3.0)
                time_score = max(0, 1 - (response_time - 1) / 10)  # Normalize around 1-2 seconds
                response_time_scores.append(time_score)
                
                # Accuracy score based on feedback
                accuracy = 1.0 if data_point.get("is_helpful", False) else 0.0
                accuracy_scores.append(accuracy)
            
            # Weighted combination of metrics
            weights = {
                "satisfaction": 0.4,
                "response_time": 0.2, 
                "accuracy": 0.4
            }
            
            final_score = (
                weights["satisfaction"] * np.mean(satisfaction_scores) +
                weights["response_time"] * np.mean(response_time_scores) +
                weights["accuracy"] * np.mean(accuracy_scores)
            )
            
            return min(1.0, max(0.0, final_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logging.error(f"Error evaluating parameters: {e}")
            return 0.5
    
    def objective_function(self, trial: optuna.Trial, performance_data: List[Dict[str, Any]]) -> float:
        """
        Objective function for Optuna optimization
        """
        try:
            # Get suggested parameters
            suggested_params = self.suggest_parameters(trial)
            
            # Evaluate performance with these parameters
            score = self.evaluate_parameters(suggested_params, performance_data)
            
            # Add penalty for extreme parameters to encourage stability
            penalty = 0
            
            # Penalize very high temperature (too random)
            if suggested_params.get("temperature", 0.7) > 0.9:
                penalty += 0.1
            
            # Penalize very low confidence threshold (too uncertain)
            if suggested_params.get("confidence_threshold", 0.8) < 0.6:
                penalty += 0.1
            
            # Penalize very high max_tokens (too verbose)
            if suggested_params.get("max_tokens", 200) > 400:
                penalty += 0.05
            
            final_score = max(0, score - penalty)
            
            # Store trial information
            trial.set_user_attr("suggested_params", suggested_params)
            trial.set_user_attr("raw_score", score)
            trial.set_user_attr("penalty", penalty)
            
            return final_score
            
        except Exception as e:
            logging.error(f"Error in objective function: {e}")
            return 0.0
    
    def optimize_parameters(self, performance_data: List[Dict[str, Any]], 
                          n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize parameters using Optuna
        """
        try:
            if not self.study or not performance_data:
                logging.warning("Cannot optimize: study not initialized or no performance data")
                return self.current_parameters
            
            # Run optimization
            def objective(trial):
                return self.objective_function(trial, performance_data)
            
            self.study.optimize(objective, n_trials=n_trials, timeout=30)  # 30 second timeout
            
            # Get best parameters
            if self.study.best_trial:
                best_params = self.study.best_trial.user_attrs.get("suggested_params", {})
                best_score = self.study.best_value
                
                # Update current parameters if improvement found
                if best_score > self._calculate_current_score(performance_data):
                    self.current_parameters.update(best_params)
                    self.current_parameters["optimization_version"] += 1
                    
                    # Record optimization
                    optimization_record = {
                        "timestamp": datetime.now().isoformat(),
                        "previous_score": self._calculate_current_score(performance_data),
                        "new_score": best_score,
                        "improvement": best_score - self._calculate_current_score(performance_data),
                        "parameters": best_params.copy(),
                        "n_trials": n_trials
                    }
                    
                    self.optimization_history.append(optimization_record)
                    self.last_optimization = datetime.now()
                    
                    logging.info(f"Parameters optimized. New score: {best_score:.3f}")
                else:
                    logging.info("No improvement found in optimization")
            
            return self.current_parameters
            
        except Exception as e:
            logging.error(f"Error optimizing parameters: {e}")
            return self.current_parameters
    
    def _calculate_current_score(self, performance_data: List[Dict[str, Any]]) -> float:
        """
        Calculate current parameter performance score
        """
        return self.evaluate_parameters(self.current_parameters, performance_data)
    
    def update_performance_metrics(self, query_type: str, performance_score: float,
                                 response_time: float, user_feedback: bool):
        """
        Update performance metrics for continuous monitoring
        """
        try:
            self.query_count += 1
            
            # Store performance data
            performance_entry = {
                "timestamp": datetime.now().isoformat(),
                "query_type": query_type,
                "performance_score": performance_score,
                "response_time": response_time,
                "user_feedback": user_feedback,
                "parameters_version": self.current_parameters.get("optimization_version", 1)
            }
            
            self.performance_metrics[query_type].append(performance_entry)
            
            # Trigger optimization if interval reached
            if self.query_count % self.optimization_interval == 0:
                self._trigger_automatic_optimization()
            
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")
    
    def _trigger_automatic_optimization(self):
        """
        Trigger automatic optimization based on recent performance
        """
        try:
            # Collect recent performance data
            recent_data = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for query_type, entries in self.performance_metrics.items():
                for entry in entries[-20:]:  # Last 20 entries per type
                    try:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time > cutoff_time:
                            recent_data.append({
                                "user_satisfaction": entry["performance_score"],
                                "response_time": entry["response_time"],
                                "is_helpful": entry["user_feedback"]
                            })
                    except (ValueError, KeyError):
                        continue
            
            if len(recent_data) >= 10:  # Need minimum data for optimization
                logging.info("Triggering automatic parameter optimization")
                self.optimize_parameters(recent_data, n_trials=15)
            else:
                logging.info("Insufficient data for optimization")
                
        except Exception as e:
            logging.error(f"Error in automatic optimization: {e}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Get insights from the optimization process
        """
        try:
            insights = {
                "current_parameters": self.current_parameters.copy(),
                "optimization_count": len(self.optimization_history),
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                "query_count": self.query_count,
                "parameter_stability": self._calculate_parameter_stability(),
                "performance_trends": self._calculate_performance_trends(),
                "optimization_effectiveness": self._calculate_optimization_effectiveness()
            }
            
            # Add study statistics if available
            if self.study:
                insights["study_statistics"] = {
                    "n_trials": len(self.study.trials),
                    "best_value": self.study.best_value if self.study.best_trial else None,
                    "optimization_direction": "maximize"
                }
            
            return insights
            
        except Exception as e:
            logging.error(f"Error getting optimization insights: {e}")
            return {"error": str(e)}
    
    def _calculate_parameter_stability(self) -> Dict[str, float]:
        """
        Calculate how stable parameters have been over optimizations
        """
        stability_scores = {}
        
        try:
            if len(self.optimization_history) < 2:
                return {"overall": 1.0}  # Perfectly stable if no changes
            
            # Calculate variance in parameter values
            for param_name in self.parameter_bounds.keys():
                values = []
                for opt_record in self.optimization_history[-10:]:  # Last 10 optimizations
                    param_value = opt_record["parameters"].get(param_name)
                    if param_value is not None:
                        values.append(param_value)
                
                if len(values) > 1:
                    # Normalize variance by parameter range
                    param_range = self.parameter_bounds[param_name][1] - self.parameter_bounds[param_name][0]
                    variance = np.var(values)
                    normalized_variance = variance / (param_range ** 2)
                    stability = max(0, 1 - normalized_variance)
                    stability_scores[param_name] = stability
            
            # Overall stability
            if stability_scores:
                stability_scores["overall"] = np.mean(list(stability_scores.values()))
            else:
                stability_scores["overall"] = 1.0
            
        except Exception as e:
            logging.error(f"Error calculating parameter stability: {e}")
            stability_scores = {"overall": 0.5}
        
        return stability_scores
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """
        Calculate performance trends over time
        """
        try:
            trends = {}
            
            # Analyze recent performance by query type
            for query_type, entries in self.performance_metrics.items():
                if len(entries) >= 5:
                    recent_scores = [entry["performance_score"] for entry in entries[-10:]]
                    
                    if len(recent_scores) > 1:
                        # Simple trend calculation
                        trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                        
                        trends[query_type] = {
                            "trend_direction": "improving" if trend_slope > 0.01 else "declining" if trend_slope < -0.01 else "stable",
                            "trend_magnitude": abs(trend_slope),
                            "recent_average": np.mean(recent_scores),
                            "sample_size": len(recent_scores)
                        }
            
            return trends
            
        except Exception as e:
            logging.error(f"Error calculating performance trends: {e}")
            return {}
    
    def _calculate_optimization_effectiveness(self) -> Dict[str, Any]:
        """
        Calculate how effective the optimizations have been
        """
        try:
            if not self.optimization_history:
                return {"effectiveness": "no_data"}
            
            improvements = [opt["improvement"] for opt in self.optimization_history if opt["improvement"] > 0]
            
            if improvements:
                return {
                    "effectiveness": "positive",
                    "average_improvement": np.mean(improvements),
                    "successful_optimizations": len(improvements),
                    "total_optimizations": len(self.optimization_history),
                    "success_rate": len(improvements) / len(self.optimization_history)
                }
            else:
                return {
                    "effectiveness": "neutral", 
                    "successful_optimizations": 0,
                    "total_optimizations": len(self.optimization_history)
                }
                
        except Exception as e:
            logging.error(f"Error calculating optimization effectiveness: {e}")
            return {"effectiveness": "error"}
    
    def get_parameter_recommendations(self, current_performance: float) -> List[str]:
        """
        Get recommendations for parameter adjustments
        """
        recommendations = []
        
        try:
            # Based on current performance, suggest improvements
            if current_performance < 0.6:
                recommendations.extend([
                    "Consider reducing temperature for more deterministic responses",
                    "Increase max_tokens to provide more detailed answers",
                    "Lower confidence threshold to provide more responses"
                ])
            elif current_performance > 0.9:
                recommendations.extend([
                    "Current performance is excellent, maintain current parameters",
                    "Consider slight temperature increase for more varied responses"
                ])
            else:
                recommendations.append("Performance is good, minor adjustments may help")
            
            # Parameter-specific recommendations
            current_temp = self.current_parameters.get("temperature", 0.7)
            if current_temp > 0.8:
                recommendations.append("High temperature detected - responses may be too random")
            elif current_temp < 0.3:
                recommendations.append("Low temperature detected - responses may be too deterministic")
            
            # Token length recommendations
            current_tokens = self.current_parameters.get("max_tokens", 200)
            if current_tokens > 300:
                recommendations.append("High token count - responses may be too verbose")
            elif current_tokens < 150:
                recommendations.append("Low token count - responses may be too brief")
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review suggested")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def export_optimization_data(self) -> Dict[str, Any]:
        """
        Export optimization data for analysis
        """
        try:
            return {
                "current_parameters": self.current_parameters,
                "optimization_history": self.optimization_history[-50:],  # Last 50 optimizations
                "performance_metrics_summary": {
                    query_type: {
                        "count": len(entries),
                        "recent_average": np.mean([e["performance_score"] for e in entries[-10:]]) if entries else 0,
                        "recent_feedback_rate": np.mean([e["user_feedback"] for e in entries[-10:]]) if entries else 0
                    }
                    for query_type, entries in self.performance_metrics.items()
                },
                "optimization_insights": self.get_optimization_insights(),
                "export_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error exporting optimization data: {e}")
            return {"error": str(e)}
    
    def reset_optimization_data(self):
        """
        Reset optimization data (use with caution)
        """
        try:
            self.optimization_history.clear()
            self.performance_metrics.clear()
            self.current_parameters = self._get_default_parameters()
            self.query_count = 0
            self.last_optimization = None
            
            # Reinitialize study
            self._initialize_study()
            
            logging.info("Optimization data reset successfully")
            
        except Exception as e:
            logging.error(f"Error resetting optimization data: {e}")
    
    def manual_parameter_update(self, new_parameters: Dict[str, Any]) -> bool:
        """
        Manually update parameters (with validation)
        """
        try:
            # Validate parameters
            for param_name, value in new_parameters.items():
                if param_name in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param_name]
                    if not (min_val <= value <= max_val):
                        logging.error(f"Parameter {param_name} value {value} out of bounds [{min_val}, {max_val}]")
                        return False
            
            # Update parameters
            self.current_parameters.update(new_parameters)
            self.current_parameters["optimization_version"] += 1
            
            # Record manual update
            manual_update = {
                "timestamp": datetime.now().isoformat(),
                "type": "manual_update",
                "parameters": new_parameters.copy(),
                "previous_version": self.current_parameters["optimization_version"] - 1,
                "new_version": self.current_parameters["optimization_version"]
            }
            
            self.optimization_history.append(manual_update)
            
            logging.info("Manual parameter update successful")
            return True
            
        except Exception as e:
            logging.error(f"Error in manual parameter update: {e}")
            return False