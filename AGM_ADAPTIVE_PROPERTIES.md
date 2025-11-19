# Adaptive Properties of AGM in Policy Optimization

## Executive Summary

The Arithmetic-Geometric Mean (AGM) algorithm offers powerful adaptive properties that can transform static policy optimization into a dynamic, self-tuning framework. This document explores how AGM's convergence patterns, iteration dynamics, and mathematical properties can be leveraged to create adaptive training systems that respond intelligently to changing training conditions.

**Key Innovation**: By monitoring AGM convergence patterns, we can extract rich signals about training dynamics and use them to automatically adapt hyperparameters, curriculum difficulty, and optimization strategies in real-time.

## 🔍 AGM Fundamentals for Adaptation

### Mathematical Foundation

The AGM algorithm iteratively refines two sequences:
```
A_{n+1} = (A_n + H_n) / 2        # Arithmetic mean
H_{n+1} = 2 × A_n × H_n / (A_n + H_n)  # Harmonic mean

lim(n→∞) A_n = lim(n→∞) H_n = √(A_0 × H_0)  # Geometric mean
```

### Adaptive Signals Available

1. **Convergence Rate**: How quickly `|A_n - H_n|` approaches zero
2. **Mean Spread**: Distance between arithmetic and harmonic means
3. **Oscillation Patterns**: Whether convergence is smooth or oscillatory
4. **Iteration Count**: How many steps needed for convergence
5. **Stability**: Variance in convergence behavior over time

## 🎯 Core Adaptive Mechanisms

### 1. Convergence Speed as Training Signal

The rate at which AGM converges reveals training dynamics:

```python
class AdaptiveAGMController:
    """Monitor AGM convergence to extract training signals."""
    
    def __init__(self):
        self.convergence_history = []
        self.stability_threshold = 1e-4
        self.adaptive_thresholds = {
            'fast_convergence': 0.7,
            'slow_convergence': 0.2,
            'instability': 0.1
        }
        
    def compute_convergence_rate(self, arithmetic: torch.Tensor, 
                               harmonic: torch.Tensor) -> float:
        """Measure how quickly AGM is converging."""
        gap = torch.abs(arithmetic - harmonic)
        relative_gap = gap / (arithmetic + harmonic + 1e-8)
        convergence_rate = 1.0 - relative_gap.mean().item()
        
        self.convergence_history.append(convergence_rate)
        return convergence_rate
    
    def adaptive_iteration_count(self, current_gap: float, 
                               training_step: int) -> int:
        """Dynamically adjust AGM iterations based on convergence needs."""
        
        if current_gap > 0.1:  # Large gap - need more iterations
            return min(20, int(10 / current_gap))
        elif current_gap < 1e-6:  # Already converged
            return 1
        else:  # Normal convergence
            base_iterations = 5
            # Adapt based on training progress
            progress_factor = min(2.0, training_step / 1000)
            return int(base_iterations * progress_factor)
    
    def detect_training_regime(self) -> str:
        """Classify current training state based on AGM patterns."""
        
        if len(self.convergence_history) < 10:
            return "initialization"
            
        recent_rates = self.convergence_history[-10:]
        mean_rate = np.mean(recent_rates)
        rate_variance = np.var(recent_rates)
        
        if mean_rate > self.adaptive_thresholds['fast_convergence']:
            return "stable_learning"
        elif mean_rate < self.adaptive_thresholds['slow_convergence']:
            return "struggling"
        elif rate_variance > 0.1:
            return "unstable"
        else:
            return "normal_learning"
```

### 2. Dynamic Mean Selection Based on Training Phase

AGM enables intelligent mean type selection:

```python
class TrainingPhaseAdaptiveAGM:
    """Adapt mean selection strategy based on detected training phase."""
    
    def __init__(self):
        self.phase_detector = TrainingPhaseDetector()
        self.strategy_history = []
        
    def select_adaptive_strategy(self, loss_history: List[float], 
                               performance_metrics: Dict[str, float],
                               agm_convergence_rate: float) -> Tuple[str, Dict]:
        """Select optimal mean strategy based on training state."""
        
        # Detect current training phase
        phase = self.phase_detector.detect_phase(
            loss_history=loss_history,
            convergence_rate=agm_convergence_rate
        )
        
        strategy_config = {}
        
        if phase == "exploration":
            # Early training - be aggressive to explore quickly
            strategy = "arithmetic_weighted_agm"
            strategy_config = {
                "arithmetic_weight": 0.8,
                "agm_iterations": 3,
                "convergence_threshold": 1e-3
            }
            
        elif phase == "exploitation": 
            # Good performance - stay balanced
            strategy = "standard_agm"
            strategy_config = {
                "agm_iterations": 5,
                "convergence_threshold": 1e-6
            }
            
        elif phase == "instability":
            # Training unstable - be conservative
            strategy = "harmonic_weighted_agm"
            strategy_config = {
                "harmonic_weight": 0.8,
                "agm_iterations": 8,
                "convergence_threshold": 1e-5
            }
            
        elif phase == "plateau":
            # Performance stuck - try different approaches
            strategy = "oscillating_agm"
            strategy_config = {
                "oscillation_period": 10,
                "exploration_boost": 0.3
            }
            
        elif phase == "convergence":
            # Near optimal - minimal adaptation
            strategy = "geometric_mean_direct"
            strategy_config = {
                "agm_iterations": 1
            }
        
        else:  # Default case
            strategy = "standard_agm"
            strategy_config = {"agm_iterations": 5}
        
        self.strategy_history.append((phase, strategy))
        return strategy, strategy_config

class TrainingPhaseDetector:
    """Detect training phases from loss patterns and AGM signals."""
    
    def __init__(self):
        self.phase_history = []
        
    def detect_phase(self, loss_history: List[float], 
                    convergence_rate: float) -> str:
        """Analyze training patterns to classify current phase."""
        
        if len(loss_history) < 20:
            return "exploration"
        
        recent_losses = loss_history[-20:]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        loss_variance = np.var(recent_losses)
        
        # Phase detection logic
        if loss_trend < -0.01 and convergence_rate > 0.6:
            return "exploitation"  # Improving + stable
        elif loss_variance > 0.1 or convergence_rate < 0.3:
            return "instability"   # High variance or slow convergence
        elif abs(loss_trend) < 0.001 and loss_variance < 0.01:
            return "plateau"       # Flat trend + low variance
        elif loss_trend < -0.001:
            return "exploitation"  # Still improving
        else:
            return "exploration"   # Default to exploration
```

### 3. AGM-Based Hyperparameter Adaptation

Use AGM signals to automatically tune training hyperparameters:

```python
class AGMHyperparameterAdapter:
    """Adapt hyperparameters based on AGM convergence patterns."""
    
    def __init__(self, base_config: Dict[str, float]):
        self.base_config = base_config
        self.adaptation_history = []
        self.smoothing_factor = 0.9
        
    def adapt_hyperparameters(self, agm_state: Dict, 
                            performance_trend: float) -> Dict[str, float]:
        """Adapt hyperparameters based on AGM convergence patterns."""
        
        # Extract AGM signals
        convergence_speed = self.compute_convergence_speed(agm_state)
        mean_spread = self.compute_mean_spread(agm_state)
        stability = self.compute_stability(agm_state)
        
        adaptations = {}
        
        # Adaptive learning rate
        lr_base = self.base_config['learning_rate']
        if convergence_speed > 0.7:  # Fast convergence
            lr_multiplier = 1.3  # Can afford higher LR
        elif convergence_speed < 0.3:  # Slow convergence  
            lr_multiplier = 0.7  # Need lower LR for stability
        elif stability < 0.5:  # Unstable
            lr_multiplier = 0.6  # Much lower for stability
        else:
            lr_multiplier = 1.0
            
        adaptations['learning_rate'] = lr_base * lr_multiplier
        
        # Adaptive clipping epsilon
        epsilon_base = self.base_config['epsilon']
        if mean_spread > 0.3:  # Large differences between means
            epsilon_multiplier = 1.4  # More aggressive clipping
        elif mean_spread < 0.05:  # Means very close
            epsilon_multiplier = 0.8  # Gentler clipping
        else:
            epsilon_multiplier = 1.0
            
        adaptations['epsilon'] = epsilon_base * epsilon_multiplier
        
        # Adaptive batch size
        batch_size_base = self.base_config.get('batch_size', 32)
        if stability > 0.8 and convergence_speed > 0.6:
            # Very stable - can handle larger batches
            batch_size_multiplier = 1.5
        elif stability < 0.3:
            # Unstable - use smaller batches
            batch_size_multiplier = 0.7
        else:
            batch_size_multiplier = 1.0
            
        adaptations['batch_size'] = int(batch_size_base * batch_size_multiplier)
        
        # Adaptive AGM iterations
        if convergence_speed < 0.2:
            adaptations['agm_iterations'] = 10  # Need more iterations
        elif convergence_speed > 0.8:
            adaptations['agm_iterations'] = 3   # Already converging fast
        else:
            adaptations['agm_iterations'] = 5   # Standard
        
        # Smooth adaptations to prevent oscillation
        self.adaptation_history.append(adaptations)
        if len(self.adaptation_history) > 1:
            prev_adaptations = self.adaptation_history[-2]
            for key in adaptations:
                if key in prev_adaptations:
                    adaptations[key] = (self.smoothing_factor * prev_adaptations[key] + 
                                      (1 - self.smoothing_factor) * adaptations[key])
        
        return adaptations
    
    def compute_convergence_speed(self, agm_state: Dict) -> float:
        """Compute AGM convergence speed from state history."""
        if len(agm_state['arithmetic_history']) < 2:
            return 0.5  # Default
            
        recent_arithmetic = agm_state['arithmetic_history'][-5:]
        recent_harmonic = agm_state['harmonic_history'][-5:]
        
        gaps = [abs(a - h) for a, h in zip(recent_arithmetic, recent_harmonic)]
        if len(gaps) < 2:
            return 0.5
            
        # Rate of gap reduction - handle both convergence and divergence
        if gaps[0] > 1e-8:  # Avoid division by very small numbers
            gap_reduction_rate = max(0.0, (gaps[0] - gaps[-1]) / gaps[0])
            convergence_rate = gap_reduction_rate
        else:
            convergence_rate = 0.5  # Default when gaps are too small to measure
        return min(1.0, convergence_rate)
    
    def compute_mean_spread(self, agm_state: Dict) -> float:
        """Compute current spread between arithmetic and harmonic means."""
        if not agm_state['arithmetic_history'] or not agm_state['harmonic_history']:
            return 0.0
            
        current_arithmetic = agm_state['arithmetic_history'][-1]
        current_harmonic = agm_state['harmonic_history'][-1]
        
        spread = abs(current_arithmetic - current_harmonic)
        relative_spread = spread / (current_arithmetic + current_harmonic + 1e-8)
        return relative_spread
    
    def compute_stability(self, agm_state: Dict) -> float:
        """Compute stability from variance in convergence patterns."""
        if len(agm_state['arithmetic_history']) < 5:
            return 0.5  # Default
            
        recent_arithmetic = agm_state['arithmetic_history'][-10:]
        recent_harmonic = agm_state['harmonic_history'][-10:]
        
        arithmetic_variance = np.var(recent_arithmetic)
        harmonic_variance = np.var(recent_harmonic)
        
        # Lower variance = higher stability
        combined_variance = arithmetic_variance + harmonic_variance
        stability = 1.0 / (1.0 + combined_variance * 100)
        return stability
```

### 4. Multi-Scale AGM Adaptation

Apply AGM at different temporal scales for comprehensive adaptation:

```python
class MultiScaleAGMFramework:
    """Use AGM signals at multiple time scales for hierarchical adaptation."""
    
    def __init__(self):
        self.short_term_agm = AGMTracker(window=10, name="short_term")     # Last 10 steps
        self.medium_term_agm = AGMTracker(window=100, name="medium_term")  # Last 100 steps  
        self.long_term_agm = AGMTracker(window=1000, name="long_term")     # Last 1000 steps
        
    def update_all_trackers(self, agm_metrics: Dict):
        """Update all temporal scale trackers."""
        self.short_term_agm.update(agm_metrics)
        self.medium_term_agm.update(agm_metrics)
        self.long_term_agm.update(agm_metrics)
    
    def compute_hierarchical_adaptation(self) -> Dict[str, str]:
        """Use AGM signals at multiple time scales for adaptation decisions."""
        
        adaptations = {}
        
        # Short-term: Immediate reaction to instability (last 10 steps)
        short_signal = self.short_term_agm.get_adaptation_signal()
        if short_signal.instability > 0.8:
            adaptations['immediate'] = "emergency_conservative_mode"
        elif short_signal.convergence_rate > 0.9:
            adaptations['immediate'] = "accelerate_slightly"
        else:
            adaptations['immediate'] = "continue_current"
            
        # Medium-term: Trend-based adaptation (last 100 steps)
        medium_signal = self.medium_term_agm.get_adaptation_signal()
        if medium_signal.trend == "improving":
            adaptations['tactical'] = "increase_exploration"
        elif medium_signal.trend == "degrading":
            adaptations['tactical'] = "increase_stability"
        elif medium_signal.trend == "oscillating":
            adaptations['tactical'] = "dampen_oscillations"
        else:
            adaptations['tactical'] = "maintain_course"
            
        # Long-term: Regime detection (last 1000 steps)
        long_signal = self.long_term_agm.get_adaptation_signal()
        if long_signal.regime == "plateau":
            adaptations['strategic'] = "curriculum_advancement"
        elif long_signal.regime == "declining":
            adaptations['strategic'] = "reset_to_checkpoint"
        elif long_signal.regime == "improving":
            adaptations['strategic'] = "maintain_strategy"
        else:
            adaptations['strategic'] = "standard_operation"
            
        return adaptations

class AGMTracker:
    """Track AGM metrics over a sliding window."""
    
    def __init__(self, window: int, name: str):
        self.window = window
        self.name = name
        self.history = {
            'convergence_rates': [],
            'mean_spreads': [],
            'iterations_used': [],
            'timestamps': []
        }
        
    def update(self, agm_metrics: Dict):
        """Add new AGM metrics to tracking history."""
        self.history['convergence_rates'].append(agm_metrics['convergence_rate'])
        self.history['mean_spreads'].append(agm_metrics['mean_spread'])
        self.history['iterations_used'].append(agm_metrics['iterations_used'])
        self.history['timestamps'].append(agm_metrics['timestamp'])
        
        # Maintain sliding window
        for key in self.history:
            if len(self.history[key]) > self.window:
                self.history[key] = self.history[key][-self.window:]
    
    def get_adaptation_signal(self) -> 'AdaptationSignal':
        """Compute adaptation signals from tracked history."""
        if not self.history['convergence_rates']:
            return AdaptationSignal()
            
        # Compute trends and patterns
        rates = self.history['convergence_rates']
        spreads = self.history['mean_spreads']
        
        # Trend analysis
        if len(rates) >= 5:
            trend = np.polyfit(range(len(rates[-5:])), rates[-5:], 1)[0]
            if trend > 0.01:
                trend_direction = "improving"
            elif trend < -0.01:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "unknown"
        
        # Instability detection
        rate_variance = np.var(rates[-min(10, len(rates)):]) if rates else 0
        instability = min(1.0, rate_variance * 100)
        
        # Convergence rate
        current_rate = rates[-1] if rates else 0.5
        
        # Regime detection (for long-term tracker)
        if self.window >= 100 and len(rates) >= 50:
            recent_mean = np.mean(rates[-50:])
            if recent_mean > 0.8:
                regime = "excellent"
            elif recent_mean > 0.6:
                regime = "good"
            elif recent_mean > 0.4:
                regime = "acceptable"
            elif recent_mean > 0.2:
                regime = "poor"
            else:
                regime = "failing"
        else:
            regime = "unknown"
        
        return AdaptationSignal(
            trend=trend_direction,
            instability=instability,
            convergence_rate=current_rate,
            regime=regime
        )

class AdaptationSignal:
    """Container for AGM-derived adaptation signals."""
    
    def __init__(self, trend="unknown", instability=0.0, 
                 convergence_rate=0.5, regime="unknown"):
        self.trend = trend
        self.instability = instability
        self.convergence_rate = convergence_rate
        self.regime = regime
```

### 5. AGM-Guided Curriculum Learning

Use AGM convergence patterns to design adaptive curricula:

```python
class AGMCurriculumController:
    """Control curriculum progression using AGM convergence signals."""
    
    def __init__(self, difficulty_levels: List[str] = None):
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard", "expert"]
        self.current_level = 0
        self.progression_history = []
        self.stability_requirement = 0.7
        self.mastery_threshold = 0.8
        
    def update_curriculum(self, agm_convergence_rate: float, 
                         success_rate: float, 
                         performance_metrics: Dict[str, float]) -> str:
        """Adapt curriculum difficulty based on AGM signals and performance."""
        
        current_difficulty = self.difficulty_levels[self.current_level]
        
        # AGM-based readiness assessment
        agm_stability = agm_convergence_rate > self.stability_requirement
        performance_mastery = success_rate > self.mastery_threshold
        
        action = "maintain_difficulty"
        
        # Fast AGM convergence + high success = ready for harder tasks
        if agm_stability and performance_mastery and self.current_level < len(self.difficulty_levels) - 1:
            self.current_level += 1
            action = "increase_difficulty"
            
        # Slow AGM convergence + low success = need easier tasks  
        elif agm_convergence_rate < 0.3 and success_rate < 0.4 and self.current_level > 0:
            self.current_level -= 1
            action = "decrease_difficulty"
            
        # AGM oscillating but reasonable performance = intermediate appropriate
        elif 0.3 <= agm_convergence_rate <= 0.6 and success_rate > 0.5:
            action = "maintain_difficulty"
            
        # Special case: very high performance but unstable AGM = consolidate
        elif success_rate > 0.9 and agm_convergence_rate < 0.5:
            action = "consolidate_current_level"
        
        # Record progression decision
        self.progression_history.append({
            'timestamp': len(self.progression_history),
            'agm_rate': agm_convergence_rate,
            'success_rate': success_rate,
            'action': action,
            'level': self.current_level
        })
        
        return action
    
    def get_curriculum_state(self) -> Dict:
        """Get current curriculum state and progression analytics."""
        return {
            'current_level': self.current_level,
            'current_difficulty': self.difficulty_levels[self.current_level],
            'total_levels': len(self.difficulty_levels),
            'progression_rate': self.compute_progression_rate(),
            'stability_trend': self.compute_stability_trend()
        }
    
    def compute_progression_rate(self) -> float:
        """Compute rate of curriculum progression."""
        if len(self.progression_history) < 10:
            return 0.0
            
        recent_actions = [entry['action'] for entry in self.progression_history[-10:]]
        increases = recent_actions.count('increase_difficulty')
        decreases = recent_actions.count('decrease_difficulty')
        
        return (increases - decreases) / 10.0
    
    def compute_stability_trend(self) -> str:
        """Analyze stability trend from AGM convergence rates."""
        if len(self.progression_history) < 5:
            return "insufficient_data"
            
        recent_rates = [entry['agm_rate'] for entry in self.progression_history[-5:]]
        rate_trend = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
        
        if rate_trend > 0.05:
            return "improving_stability"
        elif rate_trend < -0.05:
            return "decreasing_stability"
        else:
            return "stable"
```

### 6. Population-Based AGM Training

Use AGM for coordinating population-based optimization:

```python
class PopulationAGMTrainer:
    """Coordinate population training using AGM-derived strategies."""
    
    def __init__(self, population_size: int = 8):
        self.population_size = population_size
        self.population = [self.create_agent(i) for i in range(population_size)]
        self.agm_coordinator = AGMPopulationCoordinator()
        self.generation = 0
        
    def create_agent(self, agent_id: int):
        """Create a new agent with unique AGM preferences."""
        # Distribute agents across the AGM spectrum
        # Fix: Prevent division by zero when population_size = 1
        preference_ratio = agent_id / max(1, self.population_size - 1)
        
        # Map to mean type preferences
        if preference_ratio < 0.33:
            primary_mean = "harmonic"    # Conservative agents
            agm_bias = "conservative"
        elif preference_ratio < 0.67:
            primary_mean = "geometric"   # Balanced agents
            agm_bias = "balanced"
        else:
            primary_mean = "arithmetic"  # Aggressive agents
            agm_bias = "aggressive"
            
        return PolicyAgent(
            agent_id=agent_id,
            primary_mean=primary_mean,
            agm_bias=agm_bias,
            agm_sensitivity=0.1 + 0.8 * preference_ratio
        )
    
    def coordinate_population_training(self) -> Dict[str, Any]:
        """Use AGM to coordinate different agents in population."""
        
        generation_results = []
        
        # Get AGM-based preferences for each agent
        agent_preferences = self.agm_coordinator.compute_population_preferences(
            generation=self.generation,
            population_diversity=self.compute_diversity()
        )
        
        for i, (agent, preference) in enumerate(zip(self.population, agent_preferences)):
            
            # Assign AGM-derived training strategy
            if preference.exploration_needed:
                agent.set_training_strategy("arithmetic_focused", {
                    "agm_arithmetic_weight": 0.8,
                    "exploration_bonus": 0.2
                })
            elif preference.stability_needed:
                agent.set_training_strategy("harmonic_focused", {
                    "agm_harmonic_weight": 0.8,
                    "stability_bonus": 0.2
                })
            else:
                agent.set_training_strategy("agm_adaptive", {
                    "agm_iterations": 5,
                    "adaptation_rate": 0.1
                })
                
            # Train agent with assigned strategy
            agent_metrics = agent.train_step()
            
            # Extract AGM signals from training
            agm_metrics = agent.get_agm_metrics()
            
            # Update population coordinator
            self.agm_coordinator.update_agent_performance(
                agent_id=i, 
                metrics=agent_metrics,
                agm_state=agm_metrics
            )
            
            generation_results.append({
                'agent_id': i,
                'strategy': agent.current_strategy,
                'performance': agent_metrics,
                'agm_convergence': agm_metrics['convergence_rate']
            })
        
        # AGM-based population selection and evolution
        evolution_decisions = self.agm_coordinator.evolve_population(generation_results)
        self.apply_evolution_decisions(evolution_decisions)
        
        self.generation += 1
        
        return {
            'generation': self.generation,
            'population_performance': generation_results,
            'evolution_decisions': evolution_decisions,
            'diversity_metrics': self.compute_diversity()
        }
    
    def apply_evolution_decisions(self, decisions: Dict):
        """Apply AGM-guided population evolution decisions."""
        
        # Select survivors based on AGM convergence + performance
        survivor_indices = decisions['survivors']
        
        # Create new agents to replace eliminated ones
        eliminated_indices = [i for i in range(self.population_size) 
                            if i not in survivor_indices]
        
        for eliminated_idx in eliminated_indices:
            # Create new agent with AGM-guided parameters
            parent_idx = decisions['reproduction_pairs'][eliminated_idx]
            parent_agent = self.population[parent_idx]
            
            # AGM-based mutation of parent parameters
            new_agent = self.mutate_agent_with_agm(parent_agent, decisions['mutation_strength'])
            self.population[eliminated_idx] = new_agent
    
    def compute_diversity(self) -> Dict[str, float]:
        """Compute population diversity metrics."""
        mean_preferences = [agent.primary_mean for agent in self.population]
        agm_sensitivities = [agent.agm_sensitivity for agent in self.population]
        
        return {
            'mean_type_diversity': len(set(mean_preferences)) / 3.0,  # 3 possible types
            'sensitivity_variance': np.var(agm_sensitivities),
            'strategy_entropy': self.compute_strategy_entropy()
        }

class AGMPopulationCoordinator:
    """Coordinate population evolution using AGM principles."""
    
    def __init__(self):
        self.population_history = []
        self.performance_tracker = {}
        
    def compute_population_preferences(self, generation: int, 
                                     population_diversity: Dict) -> List['AgentPreference']:
        """Compute AGM-based preferences for population members."""
        
        preferences = []
        
        # Analyze population state
        diversity_level = population_diversity['mean_type_diversity']
        
        for agent_id in range(len(self.population_history)):
            if diversity_level < 0.5:  # Low diversity - encourage exploration
                exploration_needed = True
                stability_needed = False
            elif self.get_agent_performance_trend(agent_id) < 0:  # Declining performance
                exploration_needed = False
                stability_needed = True
            else:  # Balanced
                exploration_needed = False
                stability_needed = False
                
            preferences.append(AgentPreference(
                agent_id=agent_id,
                exploration_needed=exploration_needed,
                stability_needed=stability_needed,
                agm_focus="adaptive"
            ))
        
        return preferences

class AgentPreference:
    """Container for AGM-derived agent preferences."""
    
    def __init__(self, agent_id: int, exploration_needed: bool = False,
                 stability_needed: bool = False, agm_focus: str = "adaptive"):
        self.agent_id = agent_id
        self.exploration_needed = exploration_needed
        self.stability_needed = stability_needed
        self.agm_focus = agm_focus
```

### 7. AGM Early Stopping and Convergence Detection

Use AGM patterns for intelligent training termination:

```python
class AGMConvergenceDetector:
    """Detect training convergence and early stopping using AGM patterns."""
    
    def __init__(self, patience: int = 50, min_improvement: float = 1e-4):
        self.patience = patience
        self.min_improvement = min_improvement
        self.agm_convergence_history = []
        self.performance_history = []
        self.early_stop_triggers = []
        
    def should_stop_training(self, agm_metrics: Dict, 
                           performance_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if training should stop based on AGM and performance patterns."""
        
        self.agm_convergence_history.append(agm_metrics['convergence_rate'])
        self.performance_history.append(performance_metrics['primary_metric'])
        
        # Check various early stopping conditions
        
        # 1. AGM has fully converged to geometric mean
        if self.check_agm_full_convergence():
            return True, "agm_full_convergence"
            
        # 2. AGM convergence with performance plateau
        if self.check_agm_performance_plateau():
            return True, "agm_convergence_with_plateau"
            
        # 3. AGM indicates training instability
        if self.check_agm_instability():
            return True, "agm_detected_instability"
            
        # 4. AGM optimal convergence reached
        if self.check_agm_optimal_convergence():
            return True, "agm_optimal_point_reached"
            
        return False, "continue_training"
    
    def check_agm_full_convergence(self) -> bool:
        """Check if AGM has fully converged."""
        if len(self.agm_convergence_history) < 20:
            return False
            
        recent_agm = self.agm_convergence_history[-20:]
        agm_variance = np.var(recent_agm)
        agm_mean = np.mean(recent_agm)
        
        # AGM fully converged if very low variance and high convergence
        return agm_variance < 1e-6 and agm_mean > 0.99
    
    def check_agm_performance_plateau(self) -> bool:
        """Check if AGM converged but performance plateaued."""
        if len(self.performance_history) < self.patience:
            return False
            
        recent_agm = self.agm_convergence_history[-20:]
        recent_performance = self.performance_history[-self.patience:]
        
        # AGM reasonably converged
        agm_converged = np.mean(recent_agm) > 0.8 and np.var(recent_agm) < 0.01
        
        if agm_converged:
            # Check performance plateau
            performance_improvement = recent_performance[-1] - recent_performance[0]
            performance_plateau = abs(performance_improvement) < self.min_improvement
            
            return performance_plateau
        
        return False
    
    def check_agm_instability(self) -> bool:
        """Check if AGM indicates training instability."""
        if len(self.agm_convergence_history) < 30:
            return False
            
        recent_agm = self.agm_convergence_history[-30:]
        
        # High variance in AGM convergence indicates instability
        agm_variance = np.var(recent_agm)
        mean_convergence = np.mean(recent_agm)
        
        # Instability: high variance or very low convergence rates
        return agm_variance > 0.1 or mean_convergence < 0.1
    
    def check_agm_optimal_convergence(self) -> bool:
        """Check if AGM indicates optimal convergence point reached."""
        if len(self.agm_convergence_history) < 40:
            return False
            
        # Analyze AGM convergence pattern over longer window
        agm_history = self.agm_convergence_history[-40:]
        performance_history = self.performance_history[-40:]
        
        # AGM should be stable and high
        recent_agm_mean = np.mean(agm_history[-10:])
        agm_stability = np.var(agm_history[-10:]) < 0.01
        
        # Performance should be high and stable
        recent_perf_mean = np.mean(performance_history[-10:])
        perf_trend = np.polyfit(range(len(performance_history[-20:])), 
                               performance_history[-20:], 1)[0]
        
        # Optimal point: high AGM convergence, stable, minimal performance improvement
        return (recent_agm_mean > 0.85 and 
                agm_stability and 
                recent_perf_mean > 0.8 and 
                abs(perf_trend) < self.min_improvement / 10)
```

### 8. Learnable AGM Parameters

Make AGM itself adaptive through learnable parameters:

```python
class LearnableAGMOptimizer(nn.Module):
    """AGM optimizer with learnable parameters for adaptive behavior."""
    
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        
        # Learnable AGM parameters
        self.agm_temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.convergence_threshold = nn.Parameter(torch.tensor(-6.0))  # Log scale
        self.iteration_weight = nn.Parameter(torch.tensor(0.5))
        self.arithmetic_bias = nn.Parameter(torch.tensor(0.0))
        self.harmonic_bias = nn.Parameter(torch.tensor(0.0))
        
        # Tracking for analysis
        self.parameter_history = []
        
    def adaptive_agm_step(self, arithmetic: torch.Tensor, 
                         harmonic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Learnable AGM iteration with adaptive parameters."""
        
        # Temperature-controlled averaging (softmax-like)
        temp = torch.sigmoid(self.agm_temperature)
        weights = torch.softmax(torch.stack([
            torch.ones_like(arithmetic) + self.arithmetic_bias,
            torch.ones_like(harmonic) + self.harmonic_bias
        ]), dim=0)
        
        # Weighted arithmetic mean
        new_arithmetic = weights[0] * arithmetic + weights[1] * harmonic
        
        # Learnable harmonic mean computation
        beta = torch.sigmoid(self.iteration_weight)
        standard_harmonic = 2 * arithmetic * harmonic / (arithmetic + harmonic + 1e-8)
        new_harmonic = beta * standard_harmonic + (1 - beta) * harmonic
        
        return new_arithmetic, new_harmonic
    
    def get_adaptive_convergence_threshold(self) -> float:
        """Get current adaptive convergence threshold."""
        return torch.exp(self.convergence_threshold).item()
    
    def should_continue_agm(self, arithmetic: torch.Tensor, 
                           harmonic: torch.Tensor) -> bool:
        """Determine if AGM should continue based on learnable threshold."""
        gap = torch.abs(arithmetic - harmonic)
        relative_gap = gap / (arithmetic + harmonic + 1e-8)
        threshold = self.get_adaptive_convergence_threshold()
        
        return relative_gap.mean().item() > threshold
    
    def get_parameter_summary(self) -> Dict[str, float]:
        """Get current learnable parameter values."""
        return {
            'temperature': torch.sigmoid(self.agm_temperature).item(),
            'convergence_threshold': self.get_adaptive_convergence_threshold(),
            'iteration_weight': torch.sigmoid(self.iteration_weight).item(),
            'arithmetic_bias': self.arithmetic_bias.item(),
            'harmonic_bias': self.harmonic_bias.item()
        }
```

### 9. AGM-Based Uncertainty Estimation

Use AGM convergence patterns to estimate model uncertainty:

```python
class AGMUncertaintyEstimator:
    """Estimate epistemic and aleatoric uncertainty from AGM patterns."""
    
    def __init__(self):
        self.uncertainty_history = []
        self.calibration_data = []
        
    def estimate_epistemic_uncertainty(self, agm_state: Dict) -> float:
        """Estimate model uncertainty from AGM convergence patterns."""
        
        # Extract uncertainty signals from AGM
        if not agm_state['arithmetic_history'] or not agm_state['harmonic_history']:
            return 0.5  # Default moderate uncertainty
        
        # Large spread between arithmetic/harmonic = high uncertainty
        mean_spread = abs(agm_state['arithmetic_history'][-1] - 
                         agm_state['harmonic_history'][-1])
        
        # Slow convergence = high uncertainty  
        convergence_rate = self.compute_convergence_rate(agm_state)
        
        # Oscillatory behavior = high uncertainty
        oscillation_strength = self.compute_oscillation_strength(agm_state)
        
        # Combine uncertainty signals
        spread_uncertainty = min(1.0, mean_spread * 10)
        convergence_uncertainty = 1.0 - convergence_rate
        oscillation_uncertainty = oscillation_strength
        
        # Weighted combination
        total_uncertainty = (0.4 * spread_uncertainty + 
                           0.4 * convergence_uncertainty + 
                           0.2 * oscillation_uncertainty)
        
        self.uncertainty_history.append(total_uncertainty)
        return total_uncertainty
    
    def estimate_aleatoric_uncertainty(self, agm_state: Dict, 
                                     performance_variance: float) -> float:
        """Estimate data uncertainty from AGM and performance patterns."""
        
        # AGM oscillation can indicate noisy data
        oscillation = self.compute_oscillation_strength(agm_state)
        
        # High performance variance indicates noisy rewards/labels
        normalized_variance = min(1.0, performance_variance * 100)
        
        # Combine signals
        aleatoric_uncertainty = 0.6 * normalized_variance + 0.4 * oscillation
        
        return aleatoric_uncertainty
    
    def compute_convergence_rate(self, agm_state: Dict) -> float:
        """Compute AGM convergence rate from history."""
        if len(agm_state['arithmetic_history']) < 2:
            return 0.5
            
        recent_arithmetic = agm_state['arithmetic_history'][-5:]
        recent_harmonic = agm_state['harmonic_history'][-5:]
        
        gaps = [abs(a - h) for a, h in zip(recent_arithmetic, recent_harmonic)]
        if len(gaps) < 2:
            return 0.5
            
        gap_reduction = (gaps[0] - gaps[-1]) / (gaps[0] + 1e-8)
        return max(0.0, min(1.0, gap_reduction))
    
    def compute_oscillation_strength(self, agm_state: Dict) -> float:
        """Compute oscillation strength in AGM convergence."""
        if len(agm_state['arithmetic_history']) < 10:
            return 0.0
            
        recent_arithmetic = agm_state['arithmetic_history'][-10:]
        recent_harmonic = agm_state['harmonic_history'][-10:]
        
        # Compute differences between consecutive steps
        arith_diffs = [abs(recent_arithmetic[i] - recent_arithmetic[i-1]) 
                      for i in range(1, len(recent_arithmetic))]
        harm_diffs = [abs(recent_harmonic[i] - recent_harmonic[i-1]) 
                     for i in range(1, len(recent_harmonic))]
        
        # High variance in differences indicates oscillation
        arith_oscillation = np.var(arith_diffs) if arith_diffs else 0
        harm_oscillation = np.var(harm_diffs) if harm_diffs else 0
        
        combined_oscillation = (arith_oscillation + harm_oscillation) / 2
        return min(1.0, combined_oscillation * 1000)  # Scale appropriately
```

## 🚀 Comprehensive Adaptive AGM Framework

```python
class AdaptiveAGMFramework:
    """Comprehensive adaptive AGM framework integrating all adaptive properties."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize all adaptive modules
        self.adaptation_modules = {
            'convergence_controller': AdaptiveAGMController(),
            'phase_detector': TrainingPhaseAdaptiveAGM(),
            'hyperparameter_adapter': AGMHyperparameterAdapter(config['base_hyperparams']),
            'multiscale_tracker': MultiScaleAGMFramework(),
            'curriculum_controller': AGMCurriculumController(),
            'convergence_detector': AGMConvergenceDetector(),
            'uncertainty_estimator': AGMUncertaintyEstimator()
        }
        
        # Learnable AGM if requested
        if config.get('learnable_agm', False):
            self.learnable_agm = LearnableAGMOptimizer()
        else:
            self.learnable_agm = None
            
        # State tracking
        self.training_state = {
            'step': 0,
            'phase': 'initialization',
            'adaptations_made': [],
            'performance_history': []
        }
    
    def adaptive_training_step(self, batch: Dict) -> Dict[str, Any]:
        """Execute single adaptive training step using all AGM mechanisms."""
        
        # 1. Compute AGM with current settings
        agm_metrics = self.compute_agm_step(batch)
        
        # 2. Update multi-scale trackers
        self.adaptation_modules['multiscale_tracker'].update_all_trackers(agm_metrics)
        
        # 3. Analyze convergence patterns
        convergence_signal = self.adaptation_modules['convergence_controller'].detect_training_regime()
        
        # 4. Detect current training phase
        phase_info = self.adaptation_modules['phase_detector'].select_adaptive_strategy(
            loss_history=self.training_state['performance_history'],
            performance_metrics=batch['metrics'],
            agm_convergence_rate=agm_metrics['convergence_rate']
        )
        
        # 5. Multi-scale adaptation decisions
        hierarchical_adaptations = self.adaptation_modules['multiscale_tracker'].compute_hierarchical_adaptation()
        
        # 6. Adapt hyperparameters based on AGM + phase
        new_hyperparams = self.adaptation_modules['hyperparameter_adapter'].adapt_hyperparameters(
            agm_state=agm_metrics,
            performance_trend=self.compute_performance_trend()
        )
        
        # 7. Update curriculum if needed
        curriculum_action = self.adaptation_modules['curriculum_controller'].update_curriculum(
            agm_convergence_rate=agm_metrics['convergence_rate'],
            success_rate=batch['metrics'].get('success_rate', 0.5),
            performance_metrics=batch['metrics']
        )
        
        # 8. Check for early stopping
        should_stop, stop_reason = self.adaptation_modules['convergence_detector'].should_stop_training(
            agm_metrics=agm_metrics,
            performance_metrics=batch['metrics']
        )
        
        # 9. Estimate uncertainty for decision making
        epistemic_uncertainty = self.adaptation_modules['uncertainty_estimator'].estimate_epistemic_uncertainty(agm_metrics)
        aleatoric_uncertainty = self.adaptation_modules['uncertainty_estimator'].estimate_aleatoric_uncertainty(
            agm_metrics, batch['metrics'].get('variance', 0.1)
        )
        
        # 10. Update training state
        self.training_state['step'] += 1
        self.training_state['phase'] = phase_info[0] if isinstance(phase_info, tuple) else 'unknown'
        self.training_state['performance_history'].append(batch['metrics']['primary_metric'])
        
        # Compile comprehensive results
        adaptation_results = {
            'agm_metrics': agm_metrics,
            'convergence_regime': convergence_signal,
            'training_phase': phase_info,
            'hierarchical_adaptations': hierarchical_adaptations,
            'hyperparameter_adaptations': new_hyperparams,
            'curriculum_action': curriculum_action,
            'early_stopping': {'should_stop': should_stop, 'reason': stop_reason},
            'uncertainty_estimates': {
                'epistemic': epistemic_uncertainty,
                'aleatoric': aleatoric_uncertainty,
                'total': epistemic_uncertainty + aleatoric_uncertainty
            },
            'training_state': self.training_state.copy()
        }
        
        return adaptation_results
    
    def compute_agm_step(self, batch: Dict) -> Dict[str, float]:
        """Compute AGM metrics for current batch."""
        # Implementation would depend on specific model architecture
        # This is a placeholder for the actual AGM computation
        return {
            'convergence_rate': 0.7,  # Placeholder
            'mean_spread': 0.1,       # Placeholder
            'iterations_used': 5,     # Placeholder
            'timestamp': self.training_state['step']
        }
    
    def compute_performance_trend(self) -> float:
        """Compute recent performance trend."""
        if len(self.training_state['performance_history']) < 10:
            return 0.0
            
        recent_performance = self.training_state['performance_history'][-10:]
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        return trend
```

## 🎯 Implementation TODO Roadmap

### Phase 1: Core Adaptive Mechanisms 
1. Implement `AdaptiveAGMController` for basic convergence tracking
2. Add `TrainingPhaseAdaptiveAGM` for phase-based strategy selection
3. Create `AGMHyperparameterAdapter` for basic parameter adaptation
4. Integration testing with existing HMPO framework
  =====  done  =====

### Phase 2: Multi-Scale and Advanced Features 
1. Implement `MultiScaleAGMFramework` for hierarchical adaptation
2. Add `AGMCurriculumController` for adaptive curriculum learning
3. Create `AGMConvergenceDetector` for intelligent early stopping
4. Performance optimization and stability testing
  =====  done  =====

### Phase 3: Advanced Research Features 
1. Implement `LearnableAGMOptimizer` for trainable AGM parameters
2. Add `AGMUncertaintyEstimator` for uncertainty quantification
3. Create `PopulationAGMTrainer` for population-based optimization
4. Comprehensive evaluation and benchmarking
  =====  done  =====

### Phase 4: Integration and Validation 
1. Integrate all modules into `AdaptiveAGMFramework`
2. Large-scale validation on real tasks
3. Documentation and examples
4. Open-source release preparation
  =====  done  =====

## 🏁 Expected Impact

This adaptive AGM framework transforms HMPO (Check other REPO) from a static algorithm into a **dynamic, self-tuning optimization system** that:

1. **Automatically adapts** to changing training conditions
2. **Optimizes hyperparameters** based on mathematical convergence properties  
3. **Provides uncertainty estimates** for robust decision making
4. **Enables curriculum learning** guided by convergence patterns
5. **Supports population-based** optimization strategies
6. **Detects optimal stopping points** to prevent overtraining

The mathematical foundation of AGM provides theoretical sanity while the adaptive mechanisms enable practical deployment across diverse applications and scales.

---

*This document provides the complete framework for leveraging AGM's adaptive properties in policy optimization. Each component can be implemented incrementally while maintaining compatibility with the existing HMPO (other repo) codebase.* 

The HMPO codebase will be compimentary however they will be clearly marked as different for sanity as both may get rather large and I want to avoid confusion, any cross references will be resolved by having the refs in each repo so no need to fly back and forth.

## Telemetry schema and ingestion (implemented)

- Controllers consume a repo-agnostic telemetry dict validated by `agmlib.agm.telemetry.TelemetryPayload`:
  - `step: int`
  - `agm.{arithmetic_history,harmonic_history}: list[float]`
  - `rl.td.{mean,var}: float`, `rl.q.rel_change: float`, `rl.reward.{train_mean,eval_mean}: float`
  - `muon_clip.{active,clip_rate}`
- A helper `build_telemetry_from_hmpo_record(record)` maps HMPO JSONL/CSV log entries into this schema.
- The ingestion runner `scripts/hmpo_ingest_runner.py` streams HMPO logs, runs controllers, emits events, and writes controller decisions to `logs/controllers.jsonl`.