from docker.client import DockerClient
from docker.models.containers import Container
from openrlhf.utils.interface import AgentInterface
from dataclasses import dataclass

from .swebench_evaluator import (
    SweBenchEvaluator,
    BashTool,
    swe_bench_solution_evaluation_score,
)
from .agent import (
    agent_instruction_message,
    FinishTool,
    ToolCall,
    extract_tool_calls,
    call_tool,
    ToolCallResult,
    Tool,
)

Message = dict[str, str]


@dataclass
class SweBenchAgentState:
    evaluator: SweBenchEvaluator
    tools: list[Tool]
    finished: bool = False
    step: int = 0


class SweBenchEnv(AgentInterface):
    """This dummy environment is used for testing the RLHF pipeline.
    It's a simple environment where the agent is given a prompt and must respond to it.
    The reward incentivizes a short first response and a longer second response."""

    def __init__(
        self,
        *args,
        max_steps: int = 4,
        can_finish: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.can_finish = can_finish

    def init_state(self, data: dict) -> SweBenchAgentState:
        evaluator = SweBenchEvaluator(dataset=[data], max_workers=1)
        evaluator.__enter__()

        tools: list[Tool] = [BashTool(evaluator.containers[0])] if len(evaluator.contianers) > 0 else []
        if self.can_finish:
            tools.append(FinishTool())

        return SweBenchAgentState(evaluator=evaluator, tools=tools)

    def get_next_prompt(
        self, messages: list[Message], state: SweBenchAgentState
    ) -> tuple[Message, SweBenchAgentState] | None:
        assert not state.finished

        if len(messages) == 0:
            prompt = agent_instruction_message(
                prompt=state.evaluator.problem_statements[0] if len(state.evaluator.problem_statements) > 0 else "There was an error initializing the github issue.",
                tools=state.tools,
                can_finish=self.can_finish,
            )
            message = {"role": "user", "content": prompt}
            return (message, state)

        if state.step >= self.max_steps:
            state.finished = True
            return (None, state)
        state.step += 1

        assert messages[-1]["role"] == "assistant"
        tool_calls: list[ToolCall] = extract_tool_calls(messages[-1]["content"])
        tool_calls.append(ToolCall(tool_name="bash", arguments="echo hello"))

        if len(tool_calls) == 0:
            messages.append(
                {
                    "role": "user",
                    "content": "Please call at least one tool in each message.",
                }
            )

        for tool_call in tool_calls:
            tool_call_result: ToolCallResult = call_tool(
                tool_call, all_tools=state.tools
            )

            if tool_call_result.finish_tool_called:
                state.finished = True
                return (None, state)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_result.tool_name,
                    "content": tool_call_result.tool_response,
                }
            )

        last_message = messages[-1]
        messages.pop()

        return (last_message, state)

    def is_done(self, messages: list[Message], state: SweBenchAgentState) -> bool:
        return state.finished

    def get_reward(self, messages: list[Message], state: SweBenchAgentState) -> float:
        if len(state.evaluator.containers) == 0:
            return 0.0

        reward = swe_bench_solution_evaluation_score(
            container=state.evaluator.containers[0],
            datapoint=state.evaluator.dataset[0],
        )
        state.evaluator.__exit__(None, None, None)
        return reward


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    dataset = [
        datapoint
        for datapoint in dataset
        if datapoint["instance_id"] == "astropy__astropy-13033"
    ]
    assert len(dataset) == 1

    environment = SweBenchEnv(full_data=dataset, sampling_params=None, vllm_engine=None)
    state = environment.init_state(dataset[0])
    environment_2 = SweBenchEnv(
        full_data=dataset, sampling_params=None, vllm_engine=None
    )
    state_2 = environment_2.init_state(dataset[0])
    messages = []
    messages, state = environment.get_next_prompt(messages, state)
    reward = environment.get_reward(messages, state)
