        # TEAM4_AGENT_CURRICULUM_V5

        **Agent name:** TEAM4_AGENT_CURRICULUM_V5

        **Authors**
        - Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

        ## Rubric mapping
        - Alt Agent3 -- V5 dynamic-aggressive curriculum (shared policy).

        ## Description
        Curriculum with a single shared policy (no striker/goalie split) plus dynamic aggressive shaping.

        Source trial: `ray_results/PPO_curriculum_shared_V5_dynamic_aggressive/PPO_Soccer_9fc34_00000_0_2026-04-22_01-10-48`
        Checkpoint packaged: `ray_results/PPO_curriculum_shared_V5_dynamic_aggressive/PPO_Soccer_9fc34_00000_0_2026-04-22_01-10-48/checkpoint_000136/checkpoint-136`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        - `default.pth` — state_dict for the `default` policy.
        - `__init__.py` — re-exports `TeamAgent`.

