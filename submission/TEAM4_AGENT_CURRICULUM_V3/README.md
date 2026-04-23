        # TEAM4_AGENT_CURRICULUM_V3

        **Agent name:** TEAM4_AGENT_CURRICULUM_V3

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - TEAM4_AGENT_CURRICULUM_V3 submission.

        ## Description
        See training script for details.

        Source trial: `ray_results/PPO_curriculum_multiagent_V3_proximity_up/PPO_Soccer_def8b_00000_0_2026-04-22_13-44-11`
        Checkpoint packaged: `ray_results/PPO_curriculum_multiagent_V3_proximity_up/PPO_Soccer_def8b_00000_0_2026-04-22_13-44-11/checkpoint_000375/checkpoint-375`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `striker.pth` — state_dict for the `striker` policy.
- `goalie.pth` — state_dict for the `goalie` policy.
        - `__init__.py` — re-exports `TeamAgent`.

