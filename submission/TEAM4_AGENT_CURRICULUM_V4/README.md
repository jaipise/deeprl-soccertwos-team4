        # TEAM4_AGENT_CURRICULUM_V4

        **Agent name:** TEAM4_AGENT_CURRICULUM_V4

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - Alt Agent3 -- V4 aggressive-combo curriculum variant.

        ## Description
        Curriculum multiagent variant tuned for more aggressive shaping + initial-state combos.

        Source trial: `ray_results/PPO_curriculum_multiagent_V4_aggressive_combo/PPO_Soccer_4a880_00000_0_2026-04-22_00-32-38`
        Checkpoint packaged: `ray_results/PPO_curriculum_multiagent_V4_aggressive_combo/PPO_Soccer_4a880_00000_0_2026-04-22_00-32-38/checkpoint_000127/checkpoint-127`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `striker.pth` — state_dict for the `striker` policy.
- `goalie.pth` — state_dict for the `goalie` policy.
        - `__init__.py` — re-exports `TeamAgent`.

