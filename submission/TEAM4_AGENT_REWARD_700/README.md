        # TEAM4_AGENT_REWARD_700

        **Agent name:** TEAM4_AGENT_REWARD_700

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - TEAM4_AGENT_REWARD_700 submission.

        ## Description
        See training script for details.

        Source trial: `ray_results/PPO_ceia_finetune/PPO_Soccer_ceia_00000_0_2026-04-22_16-08-31`
        Checkpoint packaged: `ray_results/PPO_ceia_finetune/PPO_Soccer_ceia_00000_0_2026-04-22_16-08-31/checkpoint_000150/checkpoint-150`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `default.pth` — state_dict for the `default` policy.
        - `__init__.py` — re-exports `TeamAgent`.

