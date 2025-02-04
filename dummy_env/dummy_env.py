from openrlhf.utils.interface import AgentInterface
from typing import *

Message = Dict[str, str]
AgentState = Any

class DummyEnv(AgentInterface):
    """This dummy environment is used for testing the RLHF pipeline.
    It's a simple environment where the agent is given a prompt and must respond to it.
    The reward incentivizes a short first response and a longer second response."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_state(self, data: dict) -> AgentState:
        return (0, data["input_prompt"][0])
    
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Tuple[Message, AgentState]:
        if state[0] == 0:
            turn_1_convo = state[1]
            return turn_1_convo, (1, state[1])
        elif state[0] == 1:
            turn_2_convo = {"role": "user", "content": "Okay, but what's that times five?"}
            return turn_2_convo, (2, state[1])
        else:
            raise ValueError("DummyEnv only supports 2 usermessages")
    
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        return state[0] == 2
    
    def get_reward(self, messages: List[Message], state: AgentState) -> float:
        assert state[0] == 2
        assert messages[1]["role"] == "assistant" and messages[3]["role"] == "assistant"
        len_first_response = len(messages[1]["content"])
        len_second_response = len(messages[3]["content"])
        return float(len_first_response) * -0.01 + float(len_second_response) * 0.01
