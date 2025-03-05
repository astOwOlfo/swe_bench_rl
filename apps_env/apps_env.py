import re
import json
from random import Random
from hashlib import sha256
from dataclasses import dataclass
from beartype import beartype

from openrlhf.utils.interface import AgentInterface

from .sandbox import DockerSandbox, CompletedProcess


Message = dict[str, str]


INITIAL_PROMPT = """Please solve the following problem.
First, please thoroughly think about the solution. Then, write the solution in a ```python ... ``` block.
It is important that the block be formatted exactly this way.
Please use python.

{problem}"""


@beartype
@dataclass
class Test:
    input: str
    expected_stdout: str


@beartype
@dataclass
class TestResult:
    input: str
    expected_stdout: str
    got_stdout: str
    got_stderr: str
    passed: bool


@beartype
@dataclass
class AppsEnvState:
    data: dict
    step: int = 0
    max_fraction_tests_passed: float = 0.0
    finished: bool = False


@beartype
class AppsEnv(AgentInterface):
    def __init__(
        self,
        *args,
        max_steps: int = 4,
        truncate_test_inputs_and_outputs_length: int = 1024,
        test_timeout_seconds: int = 5,
        max_n_tests: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.max_steps = max_steps
        self.truncate_test_inputs_and_outputs_length = (
            truncate_test_inputs_and_outputs_length
        )
        self.test_timeout_seconds = test_timeout_seconds
        self.max_n_tests = 8

        self.sandbox = DockerSandbox()

    # def __del__(self):
    #     self.sandbox.cleanup()

    def init_state(self, data: dict) -> AppsEnvState:
        return AppsEnvState(data=data)

    def get_next_prompt(
        self, messages: list[Message], state: AppsEnvState
    ) -> tuple[Message | None, AppsEnvState]:
        if len(messages) == 0:
            prompt = INITIAL_PROMPT.format(problem=state.data["question"])
            message = {"role": "user", "content": prompt}
            return (message, state)

        if state.step > self.max_steps:
            state.finished = True
            return (None, state)
        state.step += 1

        assert messages[-1]["role"] == "assistant"

        code: str | None = last_backtick_python_block(messages[-1]["content"])

        if code is None:
            message = {
                "role": "user",
                "content": "Please make sure to include at least one ```python ... ``` block in your code.",
            }
            return message, state

        test_results = [
            self.run_test(code=code, test=test) for test in self.tests(state)
        ]

        n_tests_passed = len([result for result in test_results if result.passed])
        fraction_tests_passed = n_tests_passed / len(test_results)

        state.max_fraction_tests_passed = max(
            state.max_fraction_tests_passed, fraction_tests_passed
        )

        all_tests_passed = fraction_tests_passed == 1.0
        if all_tests_passed:
            state.finished = True
            return (None, state)

        prompt = self.show_test_results_and_ask_for_new_solution_prompt(test_results)
        message = {"role": "user", "content": prompt}
        return message, state

    def show_test_results_and_ask_for_new_solution_prompt(
        self, test_results: list[TestResult]
    ) -> str:
        prompt = "Your solution is incorrect - it fails some tests.\n"
        prompt += "Here are the test results.\n"
        prompt += "Note that if the inputs or outputs are too long, they are truncated, so do not be surprised if you see [TRUNCATED] in the middle of some inputs or outputs."
        prompt += "Please rewrite the solution and make it correct."

        for test_result in test_results:
            prompt += "\n\n"
            prompt += (
                "---=== TEST "
                + ("PASSED" if test_result.passed else "FAILED")
                + " ===---\n"
            )
            prompt += "\n"
            prompt += "=== INPUT ===\n"
            prompt += self.truncate(test_result.input)
            prompt += "\n"
            prompt += "=== END INPUT ===\n"
            prompt += "\n"
            prompt += "=== EXPECTED OUTPUT ===\n"
            prompt += self.truncate(test_result.expected_stdout)
            prompt += "\n"
            prompt += "=== END EXPECTED OUTPUT ===\n"
            prompt += "\n"
            prompt += "=== GOT OUTPUT ===\n"
            prompt += self.truncate(test_result.got_stdout)
            prompt += "\n"
            prompt += "=== END GOT OUTPUT ===\n"
            prompt += "\n"
            if test_result.got_stderr.strip() != "":
                prompt += "=== GOT STDERR ===\n"
                prompt += test_result.got_stderr
                prompt += "\n"
                prompt += "=== END GOT STDERR ===\n"
                prompt += "\n"

        return prompt

    def truncate(self, string: str) -> str:
        if len(string) <= self.truncate_test_inputs_and_outputs_length:
            return string

        n = self.truncate_test_inputs_and_outputs_length // 2
        return string[:n] + "\n[TRUNCATED]\n" + string[-n:]

    def run_test(self, code: str, test: Test) -> TestResult:
        output: CompletedProcess = self.execute_code(code=code, stdin=test.input)
        passed = self.strings_are_same_just_with_different_whitespaces(
            output.stdout, test.expected_stdout
        )
        return TestResult(
            input=test.input,
            expected_stdout=test.expected_stdout,
            got_stdout=output.stdout,
            got_stderr=output.stderr,
            passed=passed,
        )

    def strings_are_same_just_with_different_whitespaces(
        self, string1: str, string2: str
    ) -> bool:
        lines1: list[str] = [
            line.strip() for line in string1.splitlines() if line.strip() != ""
        ]
        lines2: list[str] = [
            line.strip() for line in string2.splitlines() if line.strip() != ""
        ]

        fields1: list[list[str]] = [line.split() for line in lines1]
        fields2: list[list[str]] = [line.split() for line in lines2]

        fields1 = tuple(tuple(fields) for fields in fields1)
        fields2 = tuple(tuple(fields) for fields in fields2)

        return fields1 == fields2

    def execute_code(self, code: str, stdin: str) -> CompletedProcess:
        print(".", end="", flush=True)
        self.sandbox.run_command(f"cat << EOF > input\n{input}\nEOF")
        self.sandbox.run_command(f"cat << EOF > solution.py\n{code}\nEOF")
        return self.sandbox.run_command(
            "cat input | python solution.py", timeout_seconds=self.test_timeout_seconds
        )

    def tests(self, state: AppsEnvState) -> list[Test]:
        inputs_and_outputs = json.loads(state.data["input_output"])

        assert isinstance(inputs_and_outputs, dict)
        assert set(inputs_and_outputs.keys()) == {"inputs", "outputs"}

        inputs = inputs_and_outputs["inputs"]
        outputs = inputs_and_outputs["outputs"]

        assert isinstance(inputs, list)
        assert isinstance(outputs, list)
        assert all(isinstance(input, str) for input in inputs)
        assert all(isinstance(output, str) for output in outputs)
        assert len(inputs) == len(outputs)

        if len(inputs) > self.max_n_tests:
            inputs, outputs = shuffle_together(
                inputs,
                outputs,
                seed=int(
                    sha256(json.dumps(inputs_and_outputs).encode()).hexdigest(), 16
                ),
            )
        inputs = inputs[: self.max_n_tests]
        outputs = outputs[: self.max_n_tests]

        return [
            Test(input=input, expected_stdout=output)
            for input, output in zip(inputs, outputs, strict=True)
        ]

    def get_reward(self, messages: list[Message], state: AppsEnvState) -> float:
        assert state.finished
        return state.max_fraction_tests_passed

    def is_done(self, messages: list[Message], state: AppsEnvState) -> bool:
        return state.finished


@beartype
def last_backtick_python_block(string: str) -> str | None:
    """Return the content of the last ```python ``` block, or None if there is no such block."""
    matches = re.findall(r"```python\s*(.*?)\s*```", string, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


@beartype
def shuffle_together(list1: list, list2: list, seed: int) -> tuple[list, list]:
    combined = list(zip(list1, list2, strict=True))
    Random(seed).shuffle(combined)
    return [x for x, y in combined], [y for x, y in combined]


@beartype
def main() -> None:
    from openai import OpenAI
    from datasets import load_dataset

    dataset = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    dataset = list(dataset)

    environment = AppsEnv(full_data=dataset, sampling_params=None, vllm_engine=None)

    environment.run_agent_loops_with_anthropic_or_openai(
        client=OpenAI(), model="gpt-4o-mini-2024-07-18"
    )
