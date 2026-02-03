#!/usr/bin/env python3
"""
W19D2: Q-Learning Starter - Team Feature Branch Activity
=========================================================

This is the BASE CODE that teams will fork and improve.
Each team member picks ONE improvement area:
  1. Learning Rate Strategies  -> modify get_learning_rate()
  2. Exploration Strategies    -> modify select_action()
  3. State Representation      -> modify create_bins()
  4. Reward Shaping           -> modify shape_reward()

Look for sections marked: # ========== MODIFY HERE ==========

Usage:
    python w19d2_starter.py                    # Train with defaults
    python w19d2_starter.py --episodes 1000   # More episodes
    python w19d2_starter.py --no-plot         # Skip live graph
    python w19d2_starter.py --evaluate        # Evaluate only (needs saved Q-table)

Output:
    - results/qtable.json      : Saved Q-table for evaluation
    - results/report.html      : Interactive results report
    - results/scores.json      : Training scores history
"""

# =============================================================================
# SECTION 0: AUTO VENV SETUP
# =============================================================================

import os
import sys
import subprocess
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv_w19d2")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REQUIREMENTS = ["gymnasium", "numpy", "matplotlib"]

def is_in_venv():
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ) or os.environ.get("W19D2_VENV_ACTIVE") == "1"

def setup_venv():
    print("=" * 60)
    print("Setting up virtual environment...")
    print("=" * 60)

    if not os.path.exists(VENV_DIR):
        print(f"Creating venv at {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    if sys.platform == "win32":
        pip_path = os.path.join(VENV_DIR, "Scripts", "pip")
        python_path = os.path.join(VENV_DIR, "Scripts", "python")
    else:
        pip_path = os.path.join(VENV_DIR, "bin", "pip")
        python_path = os.path.join(VENV_DIR, "bin", "python")

    print("Installing dependencies...")
    subprocess.run([pip_path, "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([pip_path, "install", "--quiet"] + REQUIREMENTS, check=True)
    print("Ready!\n")

    return python_path

def run_in_venv():
    python_path = setup_venv()
    env = os.environ.copy()
    env["W19D2_VENV_ACTIVE"] = "1"
    args = [python_path, __file__] + sys.argv[1:]
    result = subprocess.run(args, env=env)
    sys.exit(result.returncode)

if not is_in_venv():
    run_in_venv()

# =============================================================================
# SECTION 1: IMPORTS (after venv is active)
# =============================================================================

import json
import time
import argparse
from datetime import datetime
from collections import defaultdict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

# Student info - CHANGE THIS!
STUDENT_NAME = "Your Name"
IMPROVEMENT_AREA = "None"  # Options: "Learning Rate", "Exploration", "State Bins", "Reward Shaping"

# Random seed for reproducibility - DO NOT CHANGE for fair comparison!
RANDOM_SEED = 42

# Default hyperparameters
DEFAULT_CONFIG = {
    "learning_rate": 0.2,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "num_bins": 12,
    "num_episodes": 500,
}


# =============================================================================
# SECTION 3: Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """
    Q-Learning Agent for CartPole.

    Students modify specific methods based on their assigned improvement area.
    """

    def __init__(self, config):
        self.learning_rate = config.get("learning_rate", 0.2)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.num_bins = config.get("num_bins", 12)

        self.epsilon = self.epsilon_start
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.total_updates = 0

        # Create discretization bins
        self.bins = self.create_bins()

    # =========================================================================
    # MODIFY HERE: STATE REPRESENTATION (Member 3)
    # =========================================================================
    def create_bins(self):
        """
        Create discretization bins for state variables.

        MEMBER 3: Modify this function to improve state representation!

        Ideas to try:
          - Different number of bins (8, 16, 24)
          - Non-uniform bins (finer near center)
          - Different ranges for each variable
        """
        # ========== MODIFY HERE: BINNING STRATEGY ==========
        num_bins = self.num_bins

        return {
            "cart_pos": np.linspace(-2.4, 2.4, num_bins),
            "cart_vel": np.linspace(-3, 3, num_bins),
            "pole_angle": np.linspace(-0.21, 0.21, num_bins * 2),  # Finer for angle
            "pole_vel": np.linspace(-3, 3, num_bins),
        }

        # IDEAS:
        # 1. More bins for critical variables:
        #    "pole_angle": np.linspace(-0.21, 0.21, num_bins * 4),
        #
        # 2. Non-uniform bins (more resolution near zero):
        #    angles = np.concatenate([
        #        np.linspace(-0.21, -0.05, 8),
        #        np.linspace(-0.05, 0.05, 16),
        #        np.linspace(0.05, 0.21, 8)
        #    ])
        # =====================================================

    def discretize(self, state):
        """Convert continuous state to discrete bins."""
        cart_pos, cart_vel, pole_angle, pole_vel = state
        return (
            np.digitize(cart_pos, self.bins["cart_pos"]),
            np.digitize(cart_vel, self.bins["cart_vel"]),
            np.digitize(pole_angle, self.bins["pole_angle"]),
            np.digitize(pole_vel, self.bins["pole_vel"]),
        )

    # =========================================================================
    # MODIFY HERE: LEARNING RATE STRATEGIES (Member 1)
    # =========================================================================
    def get_learning_rate(self, state=None):
        """
        Get the learning rate for the current update.

        MEMBER 1: Modify this function to implement adaptive learning rate!

        Ideas to try:
          - Decay over time
          - Per-state learning rates
          - Scheduled decay
        """
        # ========== MODIFY HERE: LEARNING RATE STRATEGY ==========
        return self.learning_rate

        # IDEAS:
        # 1. Time-based decay:
        #    return self.learning_rate * (0.999 ** self.total_updates)
        #
        # 2. Per-state learning (need to track visit counts):
        #    if state not in self.visit_counts:
        #        self.visit_counts[state] = 0
        #    self.visit_counts[state] += 1
        #    return 1.0 / self.visit_counts[state]
        #
        # 3. Scheduled decay:
        #    episode = self.total_updates // 500
        #    if episode < 100: return 0.5
        #    if episode < 300: return 0.2
        #    return 0.05
        # =========================================================

    # =========================================================================
    # MODIFY HERE: EXPLORATION STRATEGIES (Member 2)
    # =========================================================================
    def select_action(self, state, training=True):
        """
        Select an action using exploration strategy.

        MEMBER 2: Modify this function to improve exploration!

        Ideas to try:
          - Boltzmann/softmax exploration
          - UCB (Upper Confidence Bound)
          - Different epsilon decay schedules
        """
        discrete_state = self.discretize(state)
        q_values = self.q_table[discrete_state]

        # ========== MODIFY HERE: EXPLORATION STRATEGY ==========
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 2)
        else:
            # Exploit: best action
            return np.argmax(q_values)

        # IDEAS:
        # 1. Boltzmann/Softmax exploration:
        #    temperature = 1.0
        #    exp_q = np.exp(q_values / temperature)
        #    probs = exp_q / np.sum(exp_q)
        #    return np.random.choice([0, 1], p=probs)
        #
        # 2. Linear epsilon decay (instead of exponential):
        #    # In decay_epsilon():
        #    decay_episodes = 400
        #    self.epsilon = max(
        #        self.epsilon_end,
        #        self.epsilon_start - (episode / decay_episodes) * (self.epsilon_start - self.epsilon_end)
        #    )
        # =========================================================

    # =========================================================================
    # MODIFY HERE: REWARD SHAPING (Member 4)
    # =========================================================================
    def shape_reward(self, base_reward, state, next_state, done):
        """
        Shape the reward signal to guide learning.

        MEMBER 4: Modify this function to implement reward shaping!

        Ideas to try:
          - Angle-based penalty
          - Velocity penalty
          - Position bonus

        WARNING: Be careful not to make total rewards negative!
        """
        # ========== MODIFY HERE: REWARD SHAPING ==========
        return base_reward

        # IDEAS:
        # 1. Angle-based penalty (encourage upright pole):
        #    angle_penalty = abs(state[2]) * 2
        #    return base_reward - angle_penalty
        #
        # 2. Velocity penalty (encourage smooth control):
        #    vel_penalty = abs(state[1]) * 0.1 + abs(state[3]) * 0.1
        #    return base_reward - vel_penalty
        #
        # 3. Center position bonus:
        #    pos_penalty = abs(state[0]) * 0.5
        #    return base_reward - pos_penalty
        #
        # 4. Potential-based shaping (provably safe):
        #    def potential(s): return -abs(s[2])
        #    F = self.discount_factor * potential(next_state) - potential(state)
        #    return base_reward + F
        # ===================================================

    def update(self, state, action, reward, next_state, done):
        """Update Q-value based on experience."""
        discrete_state = self.discretize(state)
        next_discrete_state = self.discretize(next_state)

        # Apply reward shaping
        shaped_reward = self.shape_reward(reward, state, next_state, done)

        # Get learning rate
        lr = self.get_learning_rate(discrete_state)

        # Q-Learning update
        old_value = self.q_table[discrete_state][action]

        if done:
            td_target = shaped_reward
        else:
            td_target = shaped_reward + self.discount_factor * np.max(self.q_table[next_discrete_state])

        self.q_table[discrete_state][action] = old_value + lr * (td_target - old_value)
        self.total_updates += 1

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_stats(self):
        """Get agent statistics."""
        return {
            "epsilon": self.epsilon,
            "states_discovered": len(self.q_table),
            "total_updates": self.total_updates,
        }

    def save(self, filepath):
        """Save Q-table to JSON."""
        # Convert defaultdict to regular dict with string keys
        q_table_serializable = {str(k): list(v) for k, v in self.q_table.items()}

        data = {
            "student_name": STUDENT_NAME,
            "improvement_area": IMPROVEMENT_AREA,
            "config": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "num_bins": self.num_bins,
            },
            "q_table": q_table_serializable,
            "stats": self.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        """Load Q-table from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore Q-table
        self.q_table = defaultdict(lambda: np.zeros(2))
        for k, v in data["q_table"].items():
            # Convert string key back to tuple
            key = eval(k)
            self.q_table[key] = np.array(v)

        self.epsilon = self.epsilon_end  # Set to minimum for evaluation


# =============================================================================
# SECTION 4: TRAINING
# =============================================================================

def train(config, show_plot=True, verbose=True):
    """Train the Q-Learning agent."""
    np.random.seed(RANDOM_SEED)

    agent = QLearningAgent(config)
    env = gym.make("CartPole-v1")

    num_episodes = config.get("num_episodes", 500)
    scores = []

    # Setup live plot
    if show_plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Q-Learning Training - {STUDENT_NAME}", fontsize=12)

        line1, = ax1.plot([], [], 'g-', alpha=0.3, label='Episode Score')
        line2, = ax1.plot([], [], 'b-', linewidth=2, label='Moving Avg (10)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_xlim(0, num_episodes)
        ax1.set_ylim(0, 550)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_xlim(0, num_episodes)
        ax2.set_ylim(0, 1.1)
        line3, = ax2.plot([], [], 'r-', label='Epsilon')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training Q-Learning Agent")
        print(f"  Student: {STUDENT_NAME}")
        print(f"  Improvement: {IMPROVEMENT_AREA}")
        print(f"  Seed: {RANDOM_SEED}")
        print(f"{'='*60}")
        print(f"\n  {'Episode':>8} | {'Score':>6} | {'Avg(10)':>8} | {'Epsilon':>8}")
        print("  " + "-" * 45)

    start_time = time.time()
    epsilons = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=RANDOM_SEED + episode)
        total_reward = 0

        for step in range(500):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, next_state, terminated or truncated)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()
        scores.append(total_reward)
        epsilons.append(agent.epsilon)

        # Update plot
        if show_plot and (episode % 5 == 0 or episode == num_episodes - 1):
            episodes = list(range(1, len(scores) + 1))
            line1.set_data(episodes, scores)

            # Moving average
            window = 10
            if len(scores) >= window:
                moving_avg = [np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))]
                line2.set_data(episodes, moving_avg)

            line3.set_data(episodes, epsilons)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        # Print progress
        if verbose and ((episode + 1) % 50 == 0 or episode < 5):
            avg_10 = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f"  {episode + 1:>8} | {total_reward:>6.0f} | {avg_10:>8.1f} | {agent.epsilon:>8.4f}")

    env.close()
    training_time = time.time() - start_time

    if show_plot:
        plt.ioff()
        plt.close()

    # Calculate results
    results = {
        "scores": scores,
        "training_time": training_time,
        "final_avg_10": float(np.mean(scores[-10:])),
        "final_avg_50": float(np.mean(scores[-50:])) if len(scores) >= 50 else float(np.mean(scores)),
        "best_score": int(max(scores)),
        "states_discovered": len(agent.q_table),
    }

    if verbose:
        print(f"\n  Training complete in {training_time:.1f}s")
        print(f"  Final Avg (last 10): {results['final_avg_10']:.1f}")
        print(f"  States discovered: {results['states_discovered']}")

    return agent, results


# =============================================================================
# SECTION 5: EVALUATION
# =============================================================================

def evaluate(agent, num_episodes=100, verbose=True):
    """
    Evaluate trained agent with epsilon=0 (no exploration).

    This is the STANDARDIZED evaluation for fair comparison.
    DO NOT MODIFY THIS FUNCTION.
    """
    np.random.seed(1000)  # Different seed for evaluation
    env = gym.make("CartPole-v1")

    scores = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluation (100 episodes, epsilon=0)")
        print(f"{'='*60}")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=1000 + episode)
        total_reward = 0

        for step in range(500):
            action = agent.select_action(state, training=False)  # No exploration
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        scores.append(total_reward)

        if verbose and (episode + 1) % 25 == 0:
            print(f"  Episode {episode + 1}/100: Mean so far = {np.mean(scores):.1f}")

    env.close()

    results = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": int(min(scores)),
        "max": int(max(scores)),
        "success_rate": float(sum(1 for s in scores if s >= 200) / len(scores) * 100),
        "perfect_rate": float(sum(1 for s in scores if s >= 500) / len(scores) * 100),
        "scores": scores,
    }

    if verbose:
        print(f"\n  Results:")
        print(f"    Mean Score:    {results['mean']:.1f} ± {results['std']:.1f}")
        print(f"    Min/Max:       {results['min']} / {results['max']}")
        print(f"    Success Rate:  {results['success_rate']:.1f}% (score ≥ 200)")
        print(f"    Perfect Rate:  {results['perfect_rate']:.1f}% (score = 500)")

    return results


# =============================================================================
# SECTION 6: HTML REPORT GENERATION
# =============================================================================

def generate_report(training_results, eval_results, config, output_path):
    """Generate interactive HTML report with Chart.js."""

    scores = training_results["scores"]
    eval_scores = eval_results["scores"]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q-Learning Results - {STUDENT_NAME}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #10b981;
            --bg-dark: #111827;
            --bg-card: #1f2937;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #059669, #10b981);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }}
        header h1 {{ margin: 0 0 10px 0; }}
        .info {{ color: rgba(255,255,255,0.8); }}
        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .card h2 {{
            color: var(--primary);
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat {{
            background: #374151;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{ color: var(--text-secondary); font-size: 0.85rem; }}
        .stat-value {{ font-size: 1.5rem; font-weight: bold; color: var(--primary); }}
        .chart-container {{ height: 300px; }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .config-table td {{
            padding: 8px;
            border-bottom: 1px solid #374151;
        }}
        .config-table td:first-child {{ color: var(--text-secondary); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Q-Learning Results</h1>
            <div class="info">
                <strong>{STUDENT_NAME}</strong> | Improvement: {IMPROVEMENT_AREA}<br>
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </header>

        <div class="card">
            <h2>Evaluation Results</h2>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-label">Mean Score</div>
                    <div class="stat-value">{eval_results["mean"]:.1f}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Std Dev</div>
                    <div class="stat-value">±{eval_results["std"]:.1f}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-value">{eval_results["success_rate"]:.0f}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Perfect Rate</div>
                    <div class="stat-value">{eval_results["perfect_rate"]:.0f}%</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Training Progress</h2>
            <div class="chart-container">
                <canvas id="trainingChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Evaluation Distribution</h2>
            <div class="chart-container">
                <canvas id="evalChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Configuration</h2>
            <table class="config-table">
                <tr><td>Learning Rate</td><td>{config["learning_rate"]}</td></tr>
                <tr><td>Discount Factor</td><td>{config["discount_factor"]}</td></tr>
                <tr><td>Epsilon Decay</td><td>{config["epsilon_decay"]}</td></tr>
                <tr><td>Num Bins</td><td>{config["num_bins"]}</td></tr>
                <tr><td>Episodes</td><td>{config["num_episodes"]}</td></tr>
                <tr><td>Training Time</td><td>{training_results["training_time"]:.1f}s</td></tr>
                <tr><td>States Discovered</td><td>{training_results["states_discovered"]}</td></tr>
            </table>
        </div>
    </div>

    <script>
        // Training chart
        const trainingCtx = document.getElementById('trainingChart').getContext('2d');
        const scores = {json.dumps(scores)};
        const episodes = scores.map((_, i) => i + 1);

        // Calculate moving average
        const movingAvg = scores.map((_, i) => {{
            const start = Math.max(0, i - 9);
            const window = scores.slice(start, i + 1);
            return window.reduce((a, b) => a + b, 0) / window.length;
        }});

        new Chart(trainingCtx, {{
            type: 'line',
            data: {{
                labels: episodes,
                datasets: [
                    {{
                        label: 'Episode Score',
                        data: scores,
                        borderColor: 'rgba(16, 185, 129, 0.3)',
                        fill: false,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Moving Avg (10)',
                        data: movingAvg,
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ min: 0, max: 550 }}
                }}
            }}
        }});

        // Evaluation histogram
        const evalCtx = document.getElementById('evalChart').getContext('2d');
        const evalScores = {json.dumps(eval_scores)};

        // Create histogram bins
        const bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
        const counts = new Array(bins.length - 1).fill(0);
        evalScores.forEach(score => {{
            for (let i = 0; i < bins.length - 1; i++) {{
                if (score >= bins[i] && score < bins[i + 1]) {{
                    counts[i]++;
                    break;
                }}
                if (i === bins.length - 2 && score >= bins[i + 1]) {{
                    counts[i]++;
                }}
            }}
        }});

        new Chart(evalCtx, {{
            type: 'bar',
            data: {{
                labels: bins.slice(0, -1).map((b, i) => `${{b}}-${{bins[i+1]}}`),
                datasets: [{{
                    label: 'Episodes',
                    data: counts,
                    backgroundColor: '#10b981'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n  Report saved to: {output_path}")


# =============================================================================
# SECTION 7: MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="W19D2 Q-Learning Starter")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plot")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only (load existing Q-table)")
    parser.add_argument("--keep-venv", action="store_true", help="Keep venv after running")
    args = parser.parse_args()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    qtable_path = os.path.join(RESULTS_DIR, "qtable.json")
    report_path = os.path.join(RESULTS_DIR, "report.html")
    scores_path = os.path.join(RESULTS_DIR, "scores.json")

    config = DEFAULT_CONFIG.copy()
    config["num_episodes"] = args.episodes

    if args.evaluate:
        # Evaluate only mode
        if not os.path.exists(qtable_path):
            print(f"Error: No Q-table found at {qtable_path}")
            print("Run training first: python w19d2_starter.py")
            return

        agent = QLearningAgent(config)
        agent.load(qtable_path)
        eval_results = evaluate(agent)
    else:
        # Train and evaluate
        agent, training_results = train(config, show_plot=not args.no_plot)

        # Save Q-table
        agent.save(qtable_path)
        print(f"\n  Q-table saved to: {qtable_path}")

        # Save scores
        with open(scores_path, 'w') as f:
            json.dump(training_results["scores"], f)

        # Evaluate
        eval_results = evaluate(agent)

        # Generate report
        generate_report(training_results, eval_results, config, report_path)

    print(f"\n{'='*60}")
    print(f"  Done! Open {report_path} in your browser to see results.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
