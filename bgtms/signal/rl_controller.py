"""
BGTMS - Reinforcement Learning Signal Controller (M1-24)
Q-Learning / DQN Agent for adaptive traffic signal phase switching and green extensions.
"""

import random
from typing import Tuple, Dict, Any, List
import numpy as np
from bgtms.core.types import ApproachState, SignalPlan, SignalPhase


class RLSignalAgent:
    """
    Q-Learning Controller for autonomous junction phase management.
    State: (discretized_q_NS, discretized_q_EW, current_phase, elapsed_bucket)
    Actions: 0 = KEEP_GREEN, 1 = SWITCH_PHASE
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.15):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.action_history: List[int] = []

    def _discretize(self, val: float, bins: List[float]) -> int:
        for idx, b in enumerate(bins):
            if val < b:
                return idx
        return len(bins)

    def get_state(self, q_ns: float, q_ew: float, phase: int, elapsed_sec: float) -> Tuple[int, int, int, int]:
        q_bins = [15.0, 35.0, 60.0]
        elapsed_bins = [15.0, 30.0, 50.0]
        return (
            self._discretize(q_ns, q_bins),
            self._discretize(q_ew, q_bins),
            phase,
            self._discretize(elapsed_sec, elapsed_bins)
        )

    def select_action(self, state: Tuple[int, int, int, int]) -> int:
        """
        Epsilon-greedy action selection:
        0 = Extend current green
        1 = Switch phase
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2)

        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = int(np.argmax(self.q_table[state]))

        self.action_history.append(action)
        return action

    def update(self, state: Tuple[int, int, int, int], action: int, reward: float,
               next_state: Tuple[int, int, int, int]):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(2)

        best_next_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        # Bellman update
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)

    def compute_reward(self, q_ns: float, q_ew: float, throughput_pcu: float) -> float:
        """
        Negative quadratic penalty on queues + throughput bonus
        """
        penalty = (q_ns**2 + q_ew**2) * 0.05
        reward = (throughput_pcu * 2.0) - penalty
        return float(reward)
