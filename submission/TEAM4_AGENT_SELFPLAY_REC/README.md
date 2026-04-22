        # TEAM4_AGENT_SELFPLAY_REC

        **Agent name:** TEAM4_AGENT_SELFPLAY_REC

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - Alt Agent1 -- recurrent self-play baseline.

        ## Description
        PPO self-play

        Source trial: `ray_results/PPO_selfplay_rec/PPO_Soccer_32674_00000_0_2026-04-21_19-24-09`
        Checkpoint packaged: `ray_results/PPO_selfplay_rec/PPO_Soccer_32674_00000_0_2026-04-21_19-24-09/checkpoint_000485/checkpoint-485`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `striker.pth` — state_dict for the `striker` policy.
- `goalie.pth` — state_dict for the `goalie` policy.
        - `__init__.py` — re-exports `TeamAgent`.

