
# Reproducibility Instructions for Sotopia Experiment

In this report, we have tried to integrate the instructions we found useful in the Sotopia documentation along with what we ended up doing and what ended up working for us.

## Dependencies and Installation

### Prerequisites
- Python 3.9+ or 3.11 (recommended)
- Docker (recommended) or Redis
- OpenAI API key

### Installation Options

#### Option 1: Simple Installation (Recommended for most users)
```bash
# Create a virtual environment
conda create -n sotopia python=3.11
conda activate sotopia

# Install Sotopia
python -m pip install sotopia
sotopia install

# Set OpenAI API key
conda env config vars set OPENAI_API_KEY=your_key
conda activate sotopia  # Reactivate to apply the environment variable
```

#### Option 2: Manual Installation (For our experiment)
```bash
# Create a virtual environment
conda create -n sotopia python=3.11
conda activate sotopia

# Install required packages
pip install sotopia==0.1.4 redis redis-om pandas numpy matplotlib seaborn

# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Data Setup

### Option 1: Using Docker (Recommended)
```bash
# Download the pre-populated Redis database
curl -L "https://huggingface.co/datasets/cmu-lti/agents_vs_script/resolve/main/dump.rdb?download=true" --output dump.rdb

# Run Redis with the dump file mounted
docker run -v $(pwd)/dump.rdb:/data/dump.rdb -p 6379:6379 redis/redis-stack
```

### Option 2: Direct Redis Installation
```bash
# Install Redis Stack Server
curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v10.focal.x86_64.tar.gz -o redis-stack-server.tar.gz
tar -xvf redis-stack-server.tar.gz

# If on Ubuntu 22, install libssl1.1
wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb

# Create directory for Redis database
mkdir -p redis-stack-server-7.2.0-v10/var/db/redis-stack

# Download and copy the dump file
curl -L "https://huggingface.co/datasets/cmu-lti/agents_vs_script/resolve/main/dump.rdb?download=true" --output dump.rdb
cp dump.rdb redis-stack-server-7.2.0-v10/var/db/redis-stack/

# Start the Redis server
./redis-stack-server-7.2.0-v10/bin/redis-stack-server --daemonize yes
```

### Verify Redis Connection
```bash
# Set the Redis connection environment variable
export REDIS_OM_URL="redis://:@localhost:6379"

# Verify the Redis connection
python -c "import redis; client = redis.Redis(host='localhost', port=6379); print(f'Connected to Redis, loaded {client.dbsize()} data points')"
```

## Running Experiments

### Basic Sotopia Demo (Simple Test)
You can quickly verify your setup with this basic example:

```python
# Save as simple_demo.py
import asyncio
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

async def main():
    await run_async_server(
        model_dict={
            "env": "gpt-4o",
            "agent1": "gpt-4o-mini",
            "agent2": "gpt-4o-mini",
        },
        sampler=UniformSampler(),
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Run with:
```bash
python simple_demo.py
```

### Agent vs. Script Mode Comparison Experiment
For our experiment comparing agent mode and script mode, save the following script as `compare_experiment.py`:

```python
#!/usr/bin/env python3
import os
import time
import subprocess
import json
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sotopia.database import EnvironmentProfile, EpisodeLog, AgentProfile

def get_environment_ids(n: int = 50) -> List[str]:
    """Get the first n environment IDs from the database."""
    env_ids = list(EnvironmentProfile.all_pks())[:n]
    print(f"Retrieved {len(env_ids)} environment IDs")
    return env_ids

def run_agent_mode(env_ids: List[str], tag: str = "agent_comparison_exp") -> str:
    """Run agent mode on the given environment IDs."""
    env_ids_str = json.dumps(env_ids)
    cmd = [
        "python", "examples/experiment_eval.py",
        "--gin_file", "sotopia_conf/generation_utils_conf/generate.gin",
        "--gin_file", "sotopia_conf/server_conf/server.gin",
        "--gin_file", "sotopia_conf/run_async_server_in_batch.gin",
        f"--gin.ENV_IDS={env_ids_str}",
        '--gin.AGENT1_MODEL="gpt-3.5-turbo"',
        '--gin.AGENT2_MODEL="gpt-3.5-turbo"',
        '--gin.BATCH_SIZE=5',
        f'--gin.TAG="{tag}"',
        f'--gin.TAG_TO_CHECK_EXISTING_EPISODES="{tag}"',
        '--gin.PUSH_TO_DB=True',
        '--gin.VERBOSE=False',
        '--gin.LITE=False'
    ]
    
    print(f"Running agent mode with command: {' '.join(cmd)}")
    subprocess.run(cmd)
    return tag

def run_script_mode(env_ids: List[str], tag: str = "script_comparison_exp") -> str:
    """Run script mode on the given environment IDs."""
    env_ids_str = json.dumps(env_ids)
    cmd = [
        "python", "examples/generate_script.py",
        "--gin_file", "sotopia_conf/generation_utils_conf/generate.gin",
        "--gin_file", "sotopia_conf/run_async_server_in_batch_script.gin",
        f"--gin.ENV_IDS={env_ids_str}",
        '--gin.SCRIPT_MODEL="gpt-3.5-turbo"',
        '--gin.BATCH_SIZE=5',
        f'--gin.TAG="{tag}"',
        '--gin.PUSH_TO_DB=True',
        '--gin.VERBOSE=False',
        '--gin.FULL_GEN=False'
    ]
    
    print(f"Running script mode with command: {' '.join(cmd)}")
    subprocess.run(cmd)
    return tag

def get_episode_data(tag: str) -> List[Dict[str, Any]]:
    """Get detailed data for episodes with the given tag."""
    episodes = list(EpisodeLog.find(EpisodeLog.tag == tag).all())
    print(f"Found {len(episodes)} episodes with tag {tag}")
    
    episode_data = []
    for episode in episodes:
        # Get environment profile
        try:
            env_profile = EnvironmentProfile.get(episode.environment)
            env_name = env_profile.scenario if hasattr(env_profile, 'scenario') else "Unknown"
        except Exception:
            env_name = "Unknown"
        
        # Get agent profiles
        agent_names = []
        for agent_id in episode.agents:
            try:
                agent_profile = AgentProfile.get(agent_id)
                agent_names.append(f"{agent_profile.first_name} {agent_profile.last_name}")
            except Exception:
                agent_names.append("Unknown Agent")
        
        # Create rewards list handling both tuple and float formats
        rewards_list = []
        for i, reward in enumerate(episode.rewards):
            # Handle cases where reward is just a float
            if isinstance(reward, float):
                rewards_list.append({
                    "agent_index": i,
                    "agent_id": episode.agents[i] if i < len(episode.agents) else None,
                    "agent_name": agent_names[i] if i < len(agent_names) else "Unknown",
                    "overall_score": reward,
                    "metrics": {"overall_score": reward}
                })
            # Handle tuple format (overall_score, metrics_dict)
            elif isinstance(reward, tuple) and len(reward) >= 2:
                rewards_list.append({
                    "agent_index": i,
                    "agent_id": episode.agents[i] if i < len(episode.agents) else None,
                    "agent_name": agent_names[i] if i < len(agent_names) else "Unknown",
                    "overall_score": reward[0],
                    "metrics": reward[1] if isinstance(reward[1], dict) else {"overall_score": reward[0]}
                })
            else:
                print(f"Warning: Unexpected reward format: {reward}")
                # Add a placeholder with default values
                rewards_list.append({
                    "agent_index": i,
                    "agent_id": episode.agents[i] if i < len(episode.agents) else None,
                    "agent_name": agent_names[i] if i < len(agent_names) else "Unknown",
                    "overall_score": 0.0,
                    "metrics": {"overall_score": 0.0}
                })
        
        # Create episode data object
        episode_obj = {
            "episode_id": episode.pk,
            "environment_id": episode.environment,
            "environment_name": env_name,
            "tag": episode.tag,
            "agents": episode.agents,
            "agent_names": agent_names,
            "models": episode.models,
            "turns": len(episode.messages),
            "rewards": rewards_list
        }
        
        episode_data.append(episode_obj)
    
    return episode_data

def analyze_results(agent_data: List[Dict[str, Any]], script_data: List[Dict[str, Any]]) -> None:
    """Analyze and compare the results from agent mode and script mode."""
    # Extract all scores and metrics
    agent_scores = []
    script_scores = []
    
    for episode in agent_data:
        for reward in episode["rewards"]:
            score_data = {
                "episode_id": episode["episode_id"],
                "environment_name": episode["environment_name"],
                "agent_name": reward["agent_name"],
                "overall_score": reward["overall_score"],
                "mode": "agent"
            }
            # Add metrics if they exist
            if "metrics" in reward and isinstance(reward["metrics"], dict):
                for metric, value in reward["metrics"].items():
                    score_data[metric] = value
            agent_scores.append(score_data)
    
    for episode in script_data:
        for reward in episode["rewards"]:
            score_data = {
                "episode_id": episode["episode_id"],
                "environment_name": episode["environment_name"],
                "agent_name": reward["agent_name"],
                "overall_score": reward["overall_score"],
                "mode": "script"
            }
            # Add metrics if they exist
            if "metrics" in reward and isinstance(reward["metrics"], dict):
                for metric, value in reward["metrics"].items():
                    score_data[metric] = value
            script_scores.append(score_data)
    
    # Check if we have scores
    if not agent_scores and not script_scores:
        print("No scores found for analysis.")
        return
    
    # Create DataFrame for analysis - handle potential missing columns
    all_scores = pd.DataFrame(agent_scores + script_scores)
    
    # Calculate summary statistics
    print("\n=== RESULTS COMPARISON ===")
    
    agent_overall = [score["overall_score"] for score in agent_scores]
    script_overall = [score["overall_score"] for score in script_scores]
    
    if agent_overall:
        print(f"Agent Mode: {len(agent_overall)} scores")
        print(f"  Average Score: {np.mean(agent_overall):.2f}")
        print(f"  Min Score: {np.min(agent_overall):.2f}")
        print(f"  Max Score: {np.max(agent_overall):.2f}")
    else:
        print("No agent scores available.")
    
    if script_overall:
        print(f"\nScript Mode: {len(script_overall)} scores")
        print(f"  Average Score: {np.mean(script_overall):.2f}")
        print(f"  Min Score: {np.min(script_overall):.2f}")
        print(f"  Max Score: {np.max(script_overall):.2f}")
    else:
        print("No script scores available.")
    
    # Analyze metrics individually
    print("\n=== METRICS COMPARISON ===")
    metrics = ["believability", "relationship", "knowledge", "secret", 
               "social_rules", "financial_and_material_benefits", "goal"]
    
    for metric in metrics:
        if metric in all_scores.columns:
            agent_metrics = all_scores[all_scores["mode"] == "agent"][metric]
            script_metrics = all_scores[all_scores["mode"] == "script"][metric]
            
            if not agent_metrics.empty and not script_metrics.empty:
                agent_metric_mean = agent_metrics.mean()
                script_metric_mean = script_metrics.mean()
                diff = script_metric_mean - agent_metric_mean
                print(f"{metric.capitalize()}: Agent={agent_metric_mean:.2f}, Script={script_metric_mean:.2f}, Diff={diff:.2f}")
    
    # Save all data to CSV and JSON
    all_scores.to_csv('comparison_results_detailed.csv', index=False)
    print("Detailed results saved to comparison_results_detailed.csv")
    
    # Save comprehensive JSON with all episode details
    comprehensive_data = {
        "agent_mode": agent_data,
        "script_mode": script_data,
        "summary": {
            "agent_mode": {
                "count": len(agent_overall),
                "mean": float(np.mean(agent_overall)) if agent_overall else 0.0,
                "min": float(np.min(agent_overall)) if agent_overall else 0.0,
                "max": float(np.max(agent_overall)) if agent_overall else 0.0
            },
            "script_mode": {
                "count": len(script_overall),
                "mean": float(np.mean(script_overall)) if script_overall else 0.0,
                "min": float(np.min(script_overall)) if script_overall else 0.0,
                "max": float(np.max(script_overall)) if script_overall else 0.0
            }
        }
    }
    
    # Add metrics comparison if data available
    metrics_comparison = {}
    for metric in metrics:
        if metric in all_scores.columns:
            agent_metrics = all_scores[all_scores["mode"] == "agent"][metric]
            script_metrics = all_scores[all_scores["mode"] == "script"][metric]
            
            if not agent_metrics.empty and not script_metrics.empty:
                metrics_comparison[metric] = {
                    "agent_mean": float(agent_metrics.mean()),
                    "script_mean": float(script_metrics.mean()),
                    "difference": float(script_metrics.mean() - agent_metrics.mean())
                }
    
    if metrics_comparison:
        comprehensive_data["summary"]["metrics_comparison"] = metrics_comparison
    
    with open('comparison_results_comprehensive.json', 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    print("Comprehensive results saved to comparison_results_comprehensive.json")
    
    # Create visualization if we have data
    if agent_overall and script_overall:
        plt.figure(figsize=(10, 6))
        plt.hist(agent_overall, alpha=0.5, label='Agent Mode')
        plt.hist(script_overall, alpha=0.5, label='Script Mode')
        plt.xlabel('Overall Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution: Agent Mode vs Script Mode')
        plt.legend()
        plt.savefig('comparison_histogram.png')
        print("Histogram saved to comparison_histogram.png")
        
        # Create metrics comparison visualization
        metrics_to_show = [m for m in metrics if m in all_scores.columns]
        if metrics_to_show:
            agent_means = []
            script_means = []
            
            for metric in metrics_to_show:
                agent_metrics = all_scores[all_scores["mode"] == "agent"][metric]
                script_metrics = all_scores[all_scores["mode"] == "script"][metric]
                
                if not agent_metrics.empty:
                    agent_means.append(agent_metrics.mean())
                else:
                    agent_means.append(0)
                    
                if not script_metrics.empty:
                    script_means.append(script_metrics.mean())
                else:
                    script_means.append(0)
            
            metrics_df = pd.DataFrame({
                'Metric': metrics_to_show,
                'Agent': agent_means,
                'Script': script_means
            })
            
            plt.figure(figsize=(12, 8))
            bar_width = 0.35
            index = np.arange(len(metrics_to_show))
            
            plt.bar(index, metrics_df['Agent'], bar_width, label='Agent Mode')
            plt.bar(index + bar_width, metrics_df['Script'], bar_width, label='Script Mode')
            
            plt.xlabel('Metrics')
            plt.ylabel('Average Score')
            plt.title('Metrics Comparison: Agent Mode vs Script Mode')
            plt.xticks(index + bar_width / 2, metrics_df['Metric'])
            plt.legend()
            plt.savefig('metrics_comparison.png')
            print("Metrics comparison saved to metrics_comparison.png")

def main():
    # Get environment IDs
    env_ids = get_environment_ids(50)
    
    # Run experiments
    agent_tag = run_agent_mode(env_ids)
    script_tag = run_script_mode(env_ids)
    
    # Give some time for database operations to complete
    print("Waiting for database operations to complete...")
    time.sleep(5)
    
    # Get and analyze results
    agent_data = get_episode_data(agent_tag)
    script_data = get_episode_data(script_tag)
    
    analyze_results(agent_data, script_data)

if __name__ == "__main__":
    main()
```

Run the experiment with:
```bash
# Make sure you're in the Sotopia project root directory
python compare_experiment.py
```

You can adjust the number of environments by modifying line 391, you can start with a testing on 10 environments `env_ids = get_environment_ids(10)` and then use more environments (e.g., 50) for more comprehensive results.

## Evaluation and Analysis

After running the experiment, you'll have several output files:

1. **comparison_histogram.png** - Visual comparison of score distributions
2. **comparison_results_detailed.csv** - Detailed CSV with all scores
3. **comparison_results_comprehensive.json** - Complete JSON data with episode information
4. **metrics_comparison.png** - Comparison of individual metrics

### Statistical Analysis

For a deeper statistical analysis, you can run:

```bash
# Install additional required packages
pip install scipy seaborn

# Run statistical analysis
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the detailed data
df = pd.read_csv('comparison_results_detailed.csv')

# Create a box plot comparison
plt.figure(figsize=(14, 8))
sns.boxplot(x='mode', y='overall_score', data=df)
plt.title('Distribution of Scores by Mode')
plt.savefig('score_boxplot.png')

# Create a violin plot to show distribution density
plt.figure(figsize=(14, 8))
sns.violinplot(x='mode', y='overall_score', data=df)
plt.title('Score Distribution Density by Mode')
plt.savefig('score_violin.png')

# Print statistical test results
agent = df[df['mode'] == 'agent']['overall_score']
script = df[df['mode'] == 'script']['overall_score']
t_stat, p_val = stats.ttest_ind(agent, script)
print(f'T-test results: t={t_stat:.3f}, p={p_val:.6f}')
print('Significant difference' if p_val < 0.05 else 'No significant difference')
"
```