from training.agents.asymmetric_sac_agent import (
    ASACAgent,
    RecurrentASACAgent,
)
from training.agents.base_agent import CrossEnvAgentWrapper
from training.agents.gail_agent import GAILAgent
from training.agents.ppo_agent import PPOAgent
from training.agents.sac_agent import SACAgent
from training.agents.idapt_agent import IDAPTAgent

MULTI_STAGE_ALGOS = {
    "idapt": IDAPTAgent,
    "asym_ac": ASACAgent,
    "rasym_ac": RecurrentASACAgent,
}


def get_multi_stage_agent_by_name(algo):
    if algo in MULTI_STAGE_ALGOS:
        return MULTI_STAGE_ALGOS[algo]
    else:
        print(MULTI_STAGE_ALGOS)
        raise ValueError("--algo %s is not supported" % algo)


def run_in(agent, env_type, **kwargs):
    """allows an agent to run in a selected env"""
    return CrossEnvAgentWrapper(agent, env_type, **kwargs)
