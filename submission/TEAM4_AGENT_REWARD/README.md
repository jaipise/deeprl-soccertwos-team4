        # TEAM4_AGENT_REWARD

        **Agent name:** TEAM4_AGENT_REWARD

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - Agent2 -- reward / observation modification (40 pts).

        ## Description
        PPO self-play with ShapedRewardWrapper; selected checkpoint_000700 by local CEIA evaluation as the best checkpoint from the 8-hour shaped run.

        Source trial: `ray_results/PPO_selfplay_shaped/PPO_Soccer_d5022_00000_0_2026-04-21_18-24-16`
        Checkpoint packaged: `ray_results/PPO_selfplay_shaped/PPO_Soccer_d5022_00000_0_2026-04-21_18-24-16/checkpoint_000700/checkpoint-700`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `default.pth` — state_dict for the `default` policy.
        - `__init__.py` — re-exports `TeamAgent`.
        Selected by evaluation: checkpoint_000700 scored 10/20 against ceia_baseline_agent in the local sweep, better than checkpoint_000870.
