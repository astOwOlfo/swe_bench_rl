import json
from dataclasses import dataclass
from beartype import beartype

from .sandbox import DockerSandbox
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from basic_agent_env import (
    BasicAgentEnv,
    BasicAgentState,
    Tool,
    Message,
)


@beartype
@dataclass
class BashTool(Tool):
    sandbox: DockerSandbox
    verbose: bool = False
    max_output_length: int = 1024

    def name(self) -> str:
        return "bash"

    def description(self) -> str:
        return "Execute bash code."

    def argument_description(self) -> str:
        return "Bash code to be executed."

    def _call(self, command: str) -> str:
        if self.verbose:
            print("\033[1;31mRUNNING COMMAND:\033[0m", command)

        result = self.sandbox.run_command(command)

        if self.verbose:
            print(f"\033[1;31mEXIT CODE: {result.returncode}\033[0m")
            print("\033[1;31mSTDOUT:\033[0m", result.stdout)
            print("\033[1;31mSTDOUT:\033[0m", result.stderr)

        return (
            f"<exit_code>{result.returncode}</exit_code>\n"
            f"<stdout>{self._truncate(result.stdout)}</stdout>\n"
            f"<stderr>{self._truncate(result.stderr)}</stderr>"
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_length:
            return text
        return (
            text[: self.max_output_length // 2]
            + "[TRUNCATED]"
            + text[-self.max_output_length // 2 :]
        )


def tests_passed(sandbox: DockerSandbox, tests: str) -> bool:
    sandbox.run_command(f"cat << EOF > tests.py\n{tests}\nEOF")
    completed_process = sandbox.run_command("pytest tests.py")
    return completed_process.returncode == 0


PUBLIC_TEST_REWARD: float = 0.2
PRIVATE_TEST_REWARD: float = 0.8


@beartype
class BashBenchEnv(BasicAgentEnv):
    def get_tools(self, data: dict) -> list[Tool]:
        sandbox = DockerSandbox()
        sandbox.run_command(f"cat << EOF > public_tests.py\n{data['public_tests']}\nEOF")
        return [BashTool(sandbox=sandbox)]

    def get_prompt(self, data: dict) -> str:
        return data["task_description"].removeprefix("Task: ") + "\n\nYou can test whether what you did is correct by running `python public_tests.py`."

    def get_reward(self, messages: list[Message], state: BasicAgentState) -> float:
        bash_tool = next(
            iter(tool for tool in state.tools if isinstance(tool, BashTool))
        )
        sandbox = bash_tool.sandbox

        public_tests_passed: bool = tests_passed(
            sandbox=sandbox, tests=state.data["public_tests"]
        )
        private_tests_passed: bool = tests_passed(
            sandbox=sandbox, tests=state.data["private_tests"]
        )

        sandbox.cleanup()

        reward = 0.0
        if public_tests_passed:
            reward += PUBLIC_TEST_REWARD
        if private_tests_passed:
            reward += PRIVATE_TEST_REWARD
        return reward


def main() -> None:
    with open("../bash-bench/valid_tasks.json", "r") as f:
        dataset = json.load(f)

    env = BashBenchEnv(dataset, None, None)
    state = env.init_state(dataset[0])
    message, state = env.get_next_prompt(messages=[], state=state)
    assert message is not None
    reward = env.get_reward(messages=[message], state=state)
    print(f"{reward=}")


def main_2() -> None:
    from vllm import LLM, SamplingParams
    import json
    from more_itertools import chunked
    from collections.abc import Callable, Iterable
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from typing import Any


    @beartype
    def delayed(function: Callable) -> Callable[..., Callable[[], Any]]:
        def workload(*args, **kwargs) -> Callable:
            return lambda: function(*args, **kwargs)

        return workload


    @beartype
    def threaded_map(
        delayed_functions: Iterable[Callable[[], Any]],
        max_workers: int,
        tqdm_description: str | None = None,
    ) -> list[Any]:
        """
        Use `threaded_map([delayed(f)(...) for ... in ...])` to run `[f(...) for ... in ...]` in a threaded way.
        """

        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     return list(executor.map(lambda f: f(), delayed_functions))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(lambda f: f(), f) for f in delayed_functions]
            results = []
            for future in tqdm(futures, total=len(futures), desc=tqdm_description):
                results.append(future.result())
        return results


    with open("/home/ubuntu/vlad/bash-bench/valid_tasks.json", "r") as f:
        whole_dataset = json.load(f)

    llm = LLM(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization="fp8",
        enable_prefix_caching=True,
        tensor_parallel_size=8,
    )

    for dataset in chunked(whole_dataset, 64):
        env = BashBenchEnv(
            full_data=dataset,
            sampling_params=None,
            vllm_engine=None,
        )

        states = threaded_map((delayed(env.init_state)(datapoint) for datapoint in dataset), max_workers=32)
        unfinished_states = states.copy()
        conversations = [[] for _ in dataset]
        unfinished_conversations = conversations.copy()
        for step in range(8):
            next_messages_and_states = threaded_map(
                (
                    delayed(env.get_next_prompt)(conversation, state)
                    for conversation, state in zip(unfinished_conversations, unfinished_states, strict=True)
                ),
                max_workers=32
            )
            finished_indices = []
            for i, (message, state) in enumerate(next_messages_and_states):
                if message is None:
                    finished_indices.append(i)
                    continue
                unfinished_conversations[i].append(message)
                unfinished_states[i] = state

            new_messages = llm.chat(
                unfinished_conversations,
                sampling_params=SamplingParams(temperature=0.6, max_tokens=4096)
            )
            for output, conversation in zip(new_messages, unfinished_conversations, strict=True):
                conversation.append({"role": "assistant", "content": output.outputs[0].text})

            unfinished_states = [state for i, state in enumerate(unfinished_states) if i not in finished_indices]
            unfinished_conversations = [conversation for i, conversation in enumerate(unfinished_conversations) if i not in finished_indices]

        rewards = threaded_map(
            (
                delayed(env.get_reward)(conversation, state)
                for conversation, state in zip(conversations, states, strict=True)
            ),
            max_workers=32
        )

        print("+" * 100)
        print("+" * 100)
        print("+" * 100)
        print("FINISHED EXECUTING AGENT LOOPS")
        for i, (conversation, reward) in enumerate(zip(conversations, rewards, strict=True)):
            print("+" * 100)
            print("+" * 100)
            print("+" * 100)
            print(f"AGENT LOOP {i + 1} ------- REWARD: {reward}")
            for message in conversation:
                print("=" * 100)
                for field, value in message.items():
                    if field == "content":
                        continue
                    print(field.upper(), value)
                print("CONTENT:", message["content"])
