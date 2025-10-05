"""
FL (Federated Learning) Manager for CivicMindAI
Simulates federated learning by collecting user feedback and adapting responses
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, deque
import statistics

class FederatedLearningManager:
    """
    Federated Learning Manager for continuous improvement of CivicMindAI
    """
    
    def __init__(self, learning_rate: float = 0.1, feedback_window: int = 100):
        self.learning_rate = learning_rate
        self.feedback_window = feedback_window
        
        # Interaction tracking
        self.interactions = deque(maxlen=1000)  # Keep last 1000 interactions
        self.feedback_data = deque(maxlen=500)  # Keep last 500 feedback entries
        
        # Model adaptation parameters
        self.response_quality_scores = defaultdict(list)
        self.department_performance = defaultdict(lambda: {"correct": 0, "total": 0})
        self.issue_type_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        self.area_coverage_quality = defaultdict(list)
        
        # Learning metrics
        self.learning_metrics = {
            "total_interactions": 0,
            "total_feedback": 0,
            "positive_feedback_rate": 0.0,
            "model_updates": 0,
            "last_update": None
        }
        
        logging.info("Federated Learning Manager initialized")
    
    def record_interaction(self, query: str, response: str, issue_type: str, 
                          area_info: Dict[str, Any] = None, department: str = "Unknown"):
        """
        Record user interaction for learning
        """
        try:
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:200],  # Limit query length for storage
                "response": response[:500],  # Limit response length
                "issue_type": issue_type,
                "department": department,
                "area": area_info.get("name", "Unknown") if area_info else "Unknown",
                "response_length": len(response),
                "query_complexity": self._assess_query_complexity(query)
            }
            
            self.interactions.append(interaction)
            self.learning_metrics["total_interactions"] += 1
            
            logging.debug(f"Recorded interaction for issue type: {issue_type}")
            
        except Exception as e:
            logging.error(f"Error recording interaction: {e}")
    
    def record_feedback(self, query: str, is_helpful: bool, 
                       issue_type: str = "unknown", department: str = "Unknown"):
        """
        Record user feedback for model improvement
        """
        try:
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:200],
                "is_helpful": is_helpful,
                "issue_type": issue_type,
                "department": department,
                "feedback_score": 1.0 if is_helpful else 0.0
            }
            
            self.feedback_data.append(feedback)
            self.learning_metrics["total_feedback"] += 1
            
            # Update department performance
            self.department_performance[department]["total"] += 1
            if is_helpful:
                self.department_performance[department]["correct"] += 1
            
            # Update issue type accuracy
            self.issue_type_accuracy[issue_type]["total"] += 1
            if is_helpful:
                self.issue_type_accuracy[issue_type]["correct"] += 1
            
            # Update response quality scores
            self.response_quality_scores[issue_type].append(1.0 if is_helpful else 0.0)
            
            # Trigger model update if we have enough feedback
            if len(self.feedback_data) % 10 == 0:  # Update every 10 feedback entries
                self._update_model_parameters()
            
            logging.debug(f"Recorded feedback: {'positive' if is_helpful else 'negative'}")
            
        except Exception as e:
            logging.error(f"Error recording feedback: {e}")
    
    def _assess_query_complexity(self, query: str) -> str:
        """
        Assess complexity of user query
        """
        try:
            word_count = len(query.split())
            
            # Multiple criteria complexity assessment
            complexity_score = 0
            
            # Length-based complexity
            if word_count > 20:
                complexity_score += 2
            elif word_count > 10:
                complexity_score += 1
            
            # Multiple entity mentions
            entities = ["water", "electricity", "garbage", "road", "transport", "zone", "ward"]
            entity_mentions = sum(1 for entity in entities if entity in query.lower())
            if entity_mentions > 2:
                complexity_score += 2
            elif entity_mentions > 1:
                complexity_score += 1
            
            # Question complexity indicators
            complex_indicators = ["how", "why", "when", "multiple", "various", "different", "compare"]
            if any(indicator in query.lower() for indicator in complex_indicators):
                complexity_score += 1
            
            # Classify complexity
            if complexity_score >= 4:
                return "high"
            elif complexity_score >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logging.warning(f"Error assessing query complexity: {e}")
            return "unknown"
    
    def _update_model_parameters(self):
        """
        Update model parameters based on recent feedback
        """
        try:
            if not self.feedback_data:
                return
            
            # Calculate recent feedback metrics
            recent_feedback = list(self.feedback_data)[-self.feedback_window:]
            
            positive_feedback = sum(1 for fb in recent_feedback if fb["is_helpful"])
            total_feedback = len(recent_feedback)
            
            if total_feedback > 0:
                self.learning_metrics["positive_feedback_rate"] = positive_feedback / total_feedback
            
            # Adaptive learning based on performance
            if self.learning_metrics["positive_feedback_rate"] < 0.7:  # If satisfaction < 70%
                self._adjust_response_strategy("improve")
            elif self.learning_metrics["positive_feedback_rate"] > 0.9:  # If satisfaction > 90%
                self._adjust_response_strategy("maintain")
            
            self.learning_metrics["model_updates"] += 1
            self.learning_metrics["last_update"] = datetime.now().isoformat()
            
            logging.info(f"Updated model parameters. Positive feedback rate: {self.learning_metrics['positive_feedback_rate']:.2f}")
            
        except Exception as e:
            logging.error(f"Error updating model parameters: {e}")
    
    def _adjust_response_strategy(self, adjustment_type: str):
        """
        Adjust response generation strategy based on feedback
        """
        try:
            if adjustment_type == "improve":
                # Identify areas needing improvement
                weak_areas = self._identify_weak_performance_areas()
                
                # Log recommendations for improvement
                for area, performance in weak_areas.items():
                    logging.info(f"Weak performance identified in {area}: {performance:.2f}")
                    
            elif adjustment_type == "maintain":
                # Current performance is good, maintain current parameters
                logging.info("Performance is satisfactory, maintaining current parameters")
            
        except Exception as e:
            logging.error(f"Error adjusting response strategy: {e}")
    
    def _identify_weak_performance_areas(self) -> Dict[str, float]:
        """
        Identify areas with weak performance based on feedback
        """
        weak_areas = {}
        
        try:
            # Check department performance
            for dept, performance in self.department_performance.items():
                if performance["total"] >= 5:  # Only consider departments with enough data
                    accuracy = performance["correct"] / performance["total"]
                    if accuracy < 0.7:  # Less than 70% accuracy
                        weak_areas[f"Department_{dept}"] = accuracy
            
            # Check issue type performance
            for issue_type, performance in self.issue_type_accuracy.items():
                if performance["total"] >= 5:
                    accuracy = performance["correct"] / performance["total"]
                    if accuracy < 0.7:
                        weak_areas[f"IssueType_{issue_type}"] = accuracy
            
        except Exception as e:
            logging.error(f"Error identifying weak areas: {e}")
        
        return weak_areas
    
    def get_adaptive_parameters(self, query: str, issue_type: str, department: str) -> Dict[str, Any]:
        """
        Get adaptive parameters for response generation based on learned patterns
        """
        try:
            parameters = {
                "temperature": 0.7,  # Default temperature
                "max_tokens": 200,   # Default max tokens
                "response_style": "standard",
                "include_escalation": True,
                "confidence_threshold": 0.8
            }
            
            # Adjust based on issue type performance
            if issue_type in self.issue_type_accuracy:
                performance = self.issue_type_accuracy[issue_type]
                if performance["total"] >= 3:
                    accuracy = performance["correct"] / performance["total"]
                    
                    if accuracy < 0.6:  # Poor performance
                        parameters["temperature"] = 0.5  # More deterministic
                        parameters["max_tokens"] = 250   # More detailed response
                        parameters["response_style"] = "detailed"
                    elif accuracy > 0.9:  # Excellent performance
                        parameters["temperature"] = 0.8  # More creative
                        parameters["response_style"] = "concise"
            
            # Adjust based on department performance
            if department in self.department_performance:
                dept_performance = self.department_performance[department]
                if dept_performance["total"] >= 3:
                    accuracy = dept_performance["correct"] / dept_performance["total"]
                    
                    if accuracy < 0.6:
                        parameters["include_escalation"] = True
                        parameters["confidence_threshold"] = 0.9  # Higher threshold for uncertain areas
            
            # Adjust based on query complexity
            complexity = self._assess_query_complexity(query)
            if complexity == "high":
                parameters["max_tokens"] = 300
                parameters["response_style"] = "comprehensive"
            elif complexity == "low":
                parameters["max_tokens"] = 150
                parameters["response_style"] = "concise"
            
            return parameters
            
        except Exception as e:
            logging.error(f"Error getting adaptive parameters: {e}")
            return {
                "temperature": 0.7,
                "max_tokens": 200,
                "response_style": "standard",
                "include_escalation": True,
                "confidence_threshold": 0.8
            }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights from the learning process
        """
        try:
            insights = {
                "overall_metrics": self.learning_metrics.copy(),
                "performance_by_department": {},
                "performance_by_issue_type": {},
                "feedback_trends": self._get_feedback_trends(),
                "recommendations": self._generate_improvement_recommendations()
            }
            
            # Calculate department performance percentages
            for dept, performance in self.department_performance.items():
                if performance["total"] > 0:
                    accuracy = performance["correct"] / performance["total"]
                    insights["performance_by_department"][dept] = {
                        "accuracy": round(accuracy * 100, 1),
                        "total_queries": performance["total"],
                        "successful_queries": performance["correct"]
                    }
            
            # Calculate issue type performance
            for issue_type, performance in self.issue_type_accuracy.items():
                if performance["total"] > 0:
                    accuracy = performance["correct"] / performance["total"]
                    insights["performance_by_issue_type"][issue_type] = {
                        "accuracy": round(accuracy * 100, 1),
                        "total_queries": performance["total"],
                        "successful_queries": performance["correct"]
                    }
            
            return insights
            
        except Exception as e:
            logging.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}
    
    def _get_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze feedback trends over time
        """
        try:
            if not self.feedback_data:
                return {"trend": "insufficient_data"}
            
            # Get recent feedback (last 50 entries)
            recent_feedback = list(self.feedback_data)[-50:]
            
            # Split into two halves to compare trends
            mid_point = len(recent_feedback) // 2
            if mid_point > 0:
                first_half = recent_feedback[:mid_point]
                second_half = recent_feedback[mid_point:]
                
                first_half_positive = sum(1 for fb in first_half if fb["is_helpful"]) / len(first_half)
                second_half_positive = sum(1 for fb in second_half if fb["is_helpful"]) / len(second_half)
                
                trend_direction = "improving" if second_half_positive > first_half_positive else "declining"
                trend_magnitude = abs(second_half_positive - first_half_positive)
                
                return {
                    "trend": trend_direction,
                    "magnitude": round(trend_magnitude, 3),
                    "current_satisfaction": round(second_half_positive, 3),
                    "previous_satisfaction": round(first_half_positive, 3)
                }
            
            return {"trend": "stable", "satisfaction": sum(1 for fb in recent_feedback if fb["is_helpful"]) / len(recent_feedback)}
            
        except Exception as e:
            logging.error(f"Error analyzing feedback trends: {e}")
            return {"trend": "error", "error": str(e)}
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """
        Generate recommendations for improving the system
        """
        recommendations = []
        
        try:
            # Check overall satisfaction rate
            if self.learning_metrics["positive_feedback_rate"] < 0.7:
                recommendations.append("Overall satisfaction is below 70%. Consider improving response accuracy and completeness.")
            
            # Check weak departments
            weak_departments = []
            for dept, performance in self.department_performance.items():
                if performance["total"] >= 5:
                    accuracy = performance["correct"] / performance["total"]
                    if accuracy < 0.6:
                        weak_departments.append(dept)
            
            if weak_departments:
                recommendations.append(f"Improve information accuracy for: {', '.join(weak_departments)}")
            
            # Check weak issue types
            weak_issues = []
            for issue_type, performance in self.issue_type_accuracy.items():
                if performance["total"] >= 5:
                    accuracy = performance["correct"] / performance["total"]
                    if accuracy < 0.6:
                        weak_issues.append(issue_type.replace('_', ' ').title())
            
            if weak_issues:
                recommendations.append(f"Enhance handling of: {', '.join(weak_issues)}")
            
            # Check feedback volume
            if self.learning_metrics["total_feedback"] < 20:
                recommendations.append("Encourage more user feedback to improve learning accuracy.")
            
            # Check interaction patterns
            if len(self.interactions) >= 50:
                recent_interactions = list(self.interactions)[-50:]
                complex_queries = sum(1 for interaction in recent_interactions 
                                    if interaction["query_complexity"] == "high")
                if complex_queries / len(recent_interactions) > 0.3:
                    recommendations.append("High proportion of complex queries detected. Consider adding more detailed response templates.")
            
            if not recommendations:
                recommendations.append("System performance is satisfactory. Continue current approach.")
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations. Manual review recommended.")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def export_learning_data(self) -> Dict[str, Any]:
        """
        Export learning data for analysis
        """
        try:
            return {
                "learning_metrics": self.learning_metrics,
                "recent_interactions": list(self.interactions)[-100:],  # Last 100 interactions
                "recent_feedback": list(self.feedback_data)[-100:],    # Last 100 feedback entries
                "department_performance": dict(self.department_performance),
                "issue_type_accuracy": dict(self.issue_type_accuracy),
                "export_timestamp": datetime.now().isoformat(),
                "total_data_points": len(self.interactions) + len(self.feedback_data)
            }
        except Exception as e:
            logging.error(f"Error exporting learning data: {e}")
            return {"error": str(e)}
    
    def reset_learning_data(self):
        """
        Reset all learning data (use with caution)
        """
        try:
            self.interactions.clear()
            self.feedback_data.clear()
            self.response_quality_scores.clear()
            self.department_performance.clear()
            self.issue_type_accuracy.clear()
            self.area_coverage_quality.clear()
            
            self.learning_metrics = {
                "total_interactions": 0,
                "total_feedback": 0,
                "positive_feedback_rate": 0.0,
                "model_updates": 0,
                "last_update": None
            }
            
            logging.info("Learning data reset successfully")
            
        except Exception as e:
            logging.error(f"Error resetting learning data: {e}")
    
    def simulate_federated_update(self) -> Dict[str, Any]:
        """
        Simulate a federated learning update across multiple nodes
        """
        try:
            # Simulate aggregating updates from multiple sources
            simulated_nodes = 3
            
            # Aggregate performance metrics
            aggregated_metrics = {
                "avg_satisfaction": self.learning_metrics["positive_feedback_rate"],
                "total_interactions": self.learning_metrics["total_interactions"],
                "nodes_participating": simulated_nodes,
                "convergence_metric": min(0.95, self.learning_metrics["positive_feedback_rate"] + 0.1)
            }
            
            # Simulate parameter updates
            parameter_updates = {
                "global_learning_rate": self.learning_rate,
                "consensus_reached": aggregated_metrics["avg_satisfaction"] > 0.8,
                "update_version": self.learning_metrics["model_updates"] + 1
            }
            
            return {
                "federated_update": parameter_updates,
                "aggregated_metrics": aggregated_metrics,
                "update_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in federated update simulation: {e}")
            return {"error": str(e)}