import os
import subprocess
from subprocess import TimeoutExpired
from shutil import copytree
from uuid import uuid4
import re
from dataclasses import dataclass
from beartype import beartype

from openrlhf.utils.interface import AgentInterface, Message


@beartype
@dataclass
class AgentState:
    datapoint: dict
    step: int = 0
    solved: bool = False

    @property
    def english_problem_statement(self) -> str:
        return self.datapoint["apps_question"]

    @property
    def lean_specification(self) -> str:
        return self.datapoint["spec"]

@beartype
@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


@beartype
def execute_lean_code(code: str, lake_directory: str = "/home/ubuntu/vlad/lean/artefacts/qa/", timeout_seconds: int = 5 * 60) -> ExecutionResult:
    lake_directory_copy = lake_directory + "/../" + str(uuid4())
    copytree(lake_directory, lake_directory_copy)

    with open(f"{lake_directory_copy}/Qa/Basic.lean", "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["lake", "build"],
            capture_output=True,
            text=True,
            env=os.environ,
            timeout=timeout_seconds,
            cwd=lake_directory_copy,
        )
    except TimeoutExpired:
        return ExecutionResult(stdout="", stderr="Timed out.", exit_code=1)

    return ExecutionResult(
        stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
    )

@beartype
class FVAppsLeanAgent(AgentInterface):
    def __init__(self, *args, max_steps: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def init_state(self, data: dict) -> AgentState:
        return AgentState(datapoint=data)

    def get_next_prompt(self, messages: list[Message], state: AgentState) -> tuple[Message | None, AgentState]:
        assert not state.solved

        if self.max_steps == 0:
            return None, state

        if len(messages) == 0:
            message = {"role": "user", "content": self.initial_message(state)}
            return message, state

        if state.step >= self.max_steps:
            return None, state
        state.step += 1

        assert messages[-1]["role"] == "assistant"
        lean_code = extract_last_lean_code_block(messages[-1]["content"])

        if lean_code is None:
            message = "Could not find a ```lean ... ``` block in your response. You must include such a block in your response. It must contain the definition of the function which is a solution to the coding problem, as well as proofs of the theorems about it that you are asked to prove."
            message = {"role": "user", "content": message}
            return message, state

        execution_result = execute_lean_code(lean_code)

        error_message = self.error_message(
            lean_code=lean_code, execution_result=execution_result, state=state
        )

        if error_message is None:
            state.solved = True
            return None, state

        error_message = {"role": "user", "content": error_message}
        return error_message, state

    def error_message(self, lean_code: str, execution_result: ExecutionResult, state: AgentState) -> str | None:
        if execution_result.exit_code != 0:
            return f"Executing lean code failed.\nSTDOUT: {execution_result.stdout}\nSTDERR: {execution_result.stderr}\nEXIT CODE: {execution_result.exit_code}"

        if "declaration uses 'sorry'" in execution_result.stdout:
            return "You are not allowed to use sorry in your solution."

        implementation_definitions = extract_definition_and_theorem_names(lean_code)
        specification_definitions = extract_definition_and_theorem_names(state.lean_specification)

        if not (set(specification_definitions) <= set(implementation_definitions)):
            return "Missing definition(s) of " + ", ".join(set(specification_definitions) - set(implementation_definitions)) + ". You must provide those definitions."

        return None

    def initial_message(self, state: AgentState) -> str:
        return f"Please consider the following coding problem:\n\n{state.english_problem_statement}\n\nYour goal is to solve write a solution to this programming problem in Lean and prove that the solution verifies some formal properties. To do this, please fill the sorries in the following code with the code that solves the programming program and proofs of theorems about it. You may add new definitions and theorems if you need auxiliary functions or lemmas, but you may not remove or rename the ones which are already provided.\n\n{state.lean_specification}"

    def is_done(self, messages: list[Message], state: AgentState) -> bool:
        return state.solved or state.step >= self.max_steps

    def get_reward(self, messages: list[Message], state: AgentState) -> float:
        return 1.0 if state.solved else 0.0


@beartype
def extract_last_lean_code_block(llm_response: str) -> str | None:
    # Find all matches of Lean code blocks
    pattern = r"```lean4?\n(.*?)\n```"
    matches = list(re.finditer(pattern, llm_response, re.DOTALL))
    
    # Return the content of the last match if any exists
    if matches:
        return matches[-1].group(1)
    return None


@beartype
def extract_definition_and_theorem_names(lean_code: str) -> list[str]:
    # pattern = r"^(?:def|theorem)\s+([a-zA-Z_'][a-zA-Z_0-9']*)[^a-zA-Z_0-9']"
    pattern = r"^(?:theorem|def)\s+([a-zA-Z_'][a-zA-Z_0-9']*)[^a-zA-Z_0-9']"
    matches = re.finditer(pattern, lean_code)
    return [match.group(1) for match in matches]


def main() -> None:
    datapoint = {
        "apps_question": "Prove that 1 + 1 = 2.",
        "spec": "theorem thm : 1 + 1 = 2 := sorry"
    }
    agent = FVAppsLeanAgent(full_data=[datapoint], sampling_params=None, vllm_engine=None)
    state = agent.init_state(datapoint)
    messages = []
    next_message, state = agent.get_next_prompt(messages, state)
    messages.append({"role": "assistant", "content": "```lean\ntheorem thm : 1 + 1 = 2 := rfl\n```"})
    next_message, state = agent.get_next_prompt(messages, state)
    if next_message is not None:
        print(next_message["content"])


def main_2() -> None:
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    dataset = list(load_dataset("quinn-dougherty/fvapps", split="train"))
    agent = FVAppsLeanAgent(
        full_data=dataset,
        vllm_engine=LLM(
            # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            model="cognitivecomputations/DeepSeek-R1-AWQ",
            # model="deepseek-ai/DeepSeek-R1",
            # model="clayray/DeepSeek-R1-AWQ_sharded",
            # quantization="fp8",
            dtype="float16",
            quantization="awq_marlin",
            max_model_len=16384,
            enable_prefix_caching=True,
            tensor_parallel_size=8,
            trust_remote_code=True,
        ),
        sampling_params=SamplingParams(max_tokens=8196, temperature=0.6)
    )

    for datapoint in dataset:
        conversation, reward = agent.run(datapoint)

        print("+" * 100)
        print("+" * 100)
        print("+" * 100)
        print(f"AGENT LOOP FINISHED ------- REWARD: {reward}")
        for message in conversation:
            print("=" * 100)
            for field, value in message.items():
                if field == "content":
                    continue
                print(field.upper(), value)
            print("CONTENT:", message["content"])


if __name__ == "__main__":
    main_2()