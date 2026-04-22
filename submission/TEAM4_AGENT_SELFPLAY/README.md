        # TEAM4_AGENT_SELFPLAY

        **Agent name:** TEAM4_AGENT_SELFPLAY

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - Agent1 -- PPO self-play baseline (same recipe as CEIA_PPOMultiagentSelfPlay).

        ## Description
        PPO with a single shared 'default' policy and a rolling 3-generation opponent archive (policy_mapping_fn: 50% current / 25% gen1 / 12.5% gen2 / 12.5% gen3). Opponents promote when episode_reward_mean > 0.5.

        Source trial: `ray_results/PPO_selfplay_baseline/PPO_Soccer_65438_00000_0_2026-04-21_17-59-40`
        Checkpoint packaged: `ray_results/PPO_selfplay_baseline/PPO_Soccer_65438_00000_0_2026-04-21_17-59-40/checkpoint_000585/checkpoint-585`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `default.pth` — state_dict for the `default` policy.
        - `__init__.py` — re-exports `TeamAgent`.

