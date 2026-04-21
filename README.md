# Soccer-Twos Starter Kit — TEAM4_AGENT Submission

**Team:** TEAM4_AGENT
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Submitted Agents

| Folder | Rubric role |
|---|---|
| `TEAM4_AGENT_BASELINE` | Agent1 — PPO self-play baseline (unmodified reward). Comparison line for the other two agents. |
| `TEAM4_AGENT_REWARD`   | Agent2 — PPO self-play + `ShapedRewardWrapper` (ball-progress reward + step penalty). Covers the reward/observation modification rubric item (40 pts). |
| `TEAM4_AGENT_SELFPLAY` | Agent3 — PPO with rolling 4-snapshot opponent archive (novel-concept +5 pts) and primary policy-performance submission (50 pts). |

Each folder contains `agent.py` (a `TeamAgent(AgentInterface)` PPO loader), `__init__.py`, its own `README.md`, and a `checkpoint/` directory populated after training.

---

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 6. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.



