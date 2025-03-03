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
        max_steps: int = 1,
        can_finish: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.can_finish = can_finish

    def init_state(self, data: dict) -> SweBenchAgentState:
        evaluator = SweBenchEvaluator(dataset=[data], max_workers=1)
        evaluator.__enter__()

        tools: list[Tool] = (
            [BashTool(evaluator.containers[0])] if len(evaluator.containers) > 0 else []
        )
        if self.can_finish:
            tools.append(FinishTool())

        return SweBenchAgentState(evaluator=evaluator, tools=tools)

    def get_next_prompt(
        self, messages: list[Message], state: SweBenchAgentState
    ) -> tuple[Message | None, SweBenchAgentState]:
        assert not state.finished

        if len(messages) == 0:
            prompt = agent_instruction_message(
                prompt=state.evaluator.problem_statements[0]
                if len(state.evaluator.problem_statements) > 0
                else "There was an error initializing the github issue.",
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
                    # "tool_call_id": tool_call_result.tool_name,
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


ALL_SWE_BENCH_REPOSITORIES = [
    "docker/compose",
    "pyca/cryptography",
    "dagster-io/dagster",
    "wagtail/wagtail",
    "qiskit/qiskit",
    "pypa/pip",
    "ray-project/ray",
    "python/typeshed",
    "johnsnowlabs/spark-nlp",
    "pandas-dev/pandas",
    "huggingface/transformers",
    "open-mmlab/mmdetection",
    "google/jax",
    "mesonbuild/meson",
    "numpy/numpy",
    "tensorflow/models",
    "tiangolo/fastapi",
    "ipython/ipython",
    "lightning-ai/lightning",
    "ytdl-org/youtube-dl",
    "datadog/integrations-core",
    "pantsbuild/pants",
    "twisted/twisted",
    "apache/mxnet",
    "prefecthq/prefect",
    "kubeflow/pipelines",
    "apache/airflow",
    "celery/celery",
    "conan-io/conan",
    "conda/conda",
    "jupyterlab/jupyterlab",
    "gitpython-developers/gitpython",
    "scipy/scipy",
    "googleapis/google-cloud-python",
    "explosion/spacy",
]


def setup_non_verified_swe_bench() -> None:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.grading import MAP_REPO_TO_PARSER
    from swebench.harness.log_parsers import parse_log_pytest

    for repository in ALL_SWE_BENCH_REPOSITORIES:
        MAP_REPO_VERSION_TO_SPECS[repository] = {
            "": {
                "python": "3.11.2",
                # "packages": "conans/requirements.txt",
                "pip_packages": [
                    "paramiko",
                    "\"docker>=7.1.0\"",
                    "\"pytest>=7, <8.0.0\"",
                    "pytest-xdist",  # To launch in N cores with pytest -n
                    "\"parameterized>=0.6.3\"",
                    "\"mock>=1.3.0, <1.4.0\"",
                    "\"WebTest>=3.0.0, <4\"",
                    "bottle",
                    "PyJWT",
                    "pluginbase",
                    "docker",
                    "setuptools",
                    "pytest-cov",
                    "\"cython<3.0.0\"",
                    # "\"PyYAML>=6.0, <7.0\"",
                    # "\"requests<3.0.0\"",
                    # '"urllib3<2.1"',
                    # '"colorama<0.5.0"',
                    # '"PyYAML<7.0"',
                    # '"patch-ng<1.19"',
                    # '"fasteners>=0.15"',
                    # '"distro<=1.8.0"',  # Quoted to handle semicolon in shell
                    # '"Jinja==1.2"',
                    # '"python-dateutil<3"',
                    # '"pytest<8.0.0"',
                    # '"pytest-xdist"',
                    # '"parameterized>=0.6.3"',
                    # '"mock<1.4.0"',
                    # '"WebTest<4"',
                    # '"bottle<0.14"',  # Combined duplicate with version
                    # '"PyJWT<3.0.0"',  # Combined duplicate with version
                    # '"pluginbase>=0.5"',  # Combined duplicate with version
                    # "docker",
                    # "setuptools",
                    # "pytest-cov",
                ],
                # "install": "pip install --only-binary :all: \"PyYAML>=6.0, <7.0\"; pip install -e .[test] --verbose",
                "install": "pip install -e .[test] --verbose",
                "test_cmd": "pytest -rA",
            }
        }
        MAP_REPO_TO_PARSER[repository] = parse_log_pytest


def main() -> None:
    from datasets import load_dataset
    from random import Random

    setup_non_verified_swe_bench()

    """
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    dataset = [
        datapoint
        for datapoint in dataset
        if datapoint["instance_id"] == "astropy__astropy-13033"
    ]
    assert len(dataset) == 1
    """

    dataset = load_dataset("princeton-nlp/SWE-bench", split="train")
    # print(f"{list(set(datapoint['repo'].lower() for datapoint in dataset))=}")
    # dataset = [list(dataset)[1234]]
    dataset = Random(42).sample(list(dataset), k=1)
    dataset = list(dataset)
    Random(42).shuffle(dataset)
    for datapoint in dataset:
        datapoint["repo"] = datapoint["repo"].lower()
        datapoint["instance_id"] = datapoint["instance_id"].lower()

    environment = SweBenchEnv(full_data=dataset, sampling_params=None, vllm_engine=None)
    for datapoint in dataset:
        print(f"{datapoint['repo']=}")
        state = environment.init_state(datapoint)
        environment_2 = SweBenchEnv(
            full_data=dataset, sampling_params=None, vllm_engine=None
        )
        state_2 = environment_2.init_state(datapoint)
        messages = []
        messages, state = environment.get_next_prompt(messages, state)
        reward = environment.get_reward(messages, state)

        exit()


def main_2() -> None:
    from vllm import LLM, SamplingParams
    import json
    from .threaded_map import threaded_map, delayed
    from more_itertools import chunked

    with open(
        "/home/ubuntu/data/swebench_verified_only_containers_which_build.json", "r"
    ) as f:
        whole_dataset = json.load(f)

    llm = LLM(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization="fp8",
        enable_prefix_caching=True,
        tensor_parallel_size=8,
    )

    for dataset in chunked(whole_dataset, 16):
        env = SweBenchEnv(
            full_data=dataset,
            sampling_params=None,
            vllm_engine=None,
        )

        states = threaded_map(
            (delayed(env.init_state)(datapoint) for datapoint in dataset),
            max_workers=32,
        )
        unfinished_states = states.copy()
        conversations = [[] for _ in dataset]
        unfinished_conversations = conversations.copy()
        for step in range(8):
            next_messages_and_states = threaded_map(
                (
                    delayed(env.get_next_prompt)(conversation, state)
                    for conversation, state in zip(
                        unfinished_conversations, unfinished_states, strict=True
                    )
                ),
                max_workers=32,
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
                sampling_params=SamplingParams(temperature=0.6, max_tokens=4096),
            )
            for output, conversation in zip(
                new_messages, unfinished_conversations, strict=True
            ):
                conversation.append(
                    {"role": "assistant", "content": output.outputs[0].text}
                )

            unfinished_states = [
                state
                for i, state in enumerate(unfinished_states)
                if i not in finished_indices
            ]
            unfinished_conversations = [
                conversation
                for i, conversation in enumerate(unfinished_conversations)
                if i not in finished_indices
            ]

        rewards = threaded_map(
            (
                delayed(env.get_reward)(conversation, state)
                for conversation, state in zip(conversations, states, strict=True)
            ),
            max_workers=32,
        )

        print("+" * 100)
        print("+" * 100)
        print("+" * 100)
        print("FINISHED EXECUTING AGENT LOOPS")
        for i, (conversation, reward) in enumerate(
            zip(conversations, rewards, strict=True)
        ):
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
