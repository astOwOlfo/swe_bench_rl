from swebench.harness.docker_build import build_instance_images, build_container
from inspect_evals.swe_bench.scorers import (
    get_eval_script,
    get_score_and_explanation_from_test_output,
)
from inspect_ai.solver import TaskState
from docker.client import DockerClient
from docker.models.containers import Container
from datasets import load_dataset
from uuid import uuid4
import logging
import json
import base64
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, ContextManager
from beartype import beartype

from .threaded_map import delayed, threaded_map
from .agent import Tool


@beartype
class SweBenchEvaluator(ContextManager):
    """
    Used to evaluate whether a solution to a SWEBench problem is correct.
    It builds docker sandboxes that allow one to interact with github repos through a terminal.
    One should resolve a given github issue to solve a SWE bench problem.

    Use the following way:

    ```
    # Initialize the environments.
    # This should be used with a context manager because __exit__ cleans up the docker sandboxes.
    # Some containers fail to build. They are excluded, as well as the corresponding dataset samples and problem statements.
    with SWEBenchEvaluator(...) as evaluator:
        # Get the problem statement of the ith datapoint.
        prompt = evaluator.prompts[i]

        # Execute a command in the sandbox environment of the ith datapoint.
        # The present working directory is by default the root directory of the github repo.
        # To solve the problem, run commands that would make modifications that resolve the github issue.
        # It returns an instance of `ExecutionResult` with attributes exit_code: bool and output: str, containing both stdout and stderr.
        output = evaluator.containers[i].exec_run(["echo", "hello"])
        output = evaluator.containers[i].exec_run("echo bonjour")

        # Note: To make an agent interact with the environments, pass `BashTool(self.containers[i])` to the agent.

        # Run this after finishing resolving all the github issues.
        # This returns a list of scores, one per github issue, which are 1.0 if it has been resolved correctly and 0.0 if it hasn't.
        # Note: I'm not sure if they can be something other than 0.0 and 1.0 and if it's necessarily 1.0 if the issue has been resolved.
        # TO DO: check this ^
        scores: list[float] = evaluator.solution_evaluation_scores()
    ```
    """

    docker_client: DockerClient
    """The docker client that was used for creating the environments."""

    containers: list[Container]
    """Docker containers with the SWE Bench environments. At initialization, the containers whose builds fail would be excluded from this list."""

    dataset: list[dict]
    """Dataset that should in the same format as princeton-nlp/SWE-bench on huggingface. At initialization, the fields corresponding to containers that fail to build will be excluded from this list."""

    problem_statements: list[str]
    """Github issue descriptions in English. At initialization, the problem statements corresponding to containers whose builds fail will be excluded from this list."""

    max_workers: int
    """Number of threads to use to parallelize docker operations."""

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        max_workers: int = 1,
    ) -> None:
        """
        See the docstring of the class for documentation.

        Arguments:
        :dataset: Dataset in the same format as princeton-nlp/SWE-bench on huggingface. Note that if the docker containers for some datapoints in it fail to build, those datapoints will be excluded.
        :max_workers: Number of threads to use to parallelize docker operations.
        """

        self.dataset = dataset
        self.max_workers = max_workers
        self.problem_statements = [
            datapoint["problem_statement"] for datapoint in self.dataset
        ]

        self.docker_client = DockerClient.from_env()

        self._build_containers()

        self._start_containers()

    def _build_containers(self) -> None:
        """
        print("WARNING: REMOVING ALL EXISTING DOCKER CONTAINERS!!!!")
        for container in self.docker_client.containers.list(all=True):
            is_swebench_image = len(container.image.tags) > 0 and all(
                tag.startswith("sweb.") for tag in container.image.tags
            )
            if is_swebench_image:
                container.remove(force=True)
        """

        unique_identifier = "---" + str(uuid4())
        successfully_built, failed_build = build_instance_images(
            client=self.docker_client,
            dataset=[
                datapoint
                | {"instance_id": datapoint["instance_id"] + unique_identifier}
                for datapoint in self.dataset
            ],  # type: ignore
            force_rebuild=False,
            max_workers=self.max_workers,
        )

        print(
            f"{len(successfully_built)} images built successfully, failed building {len(failed_build)} images"
        )

        if len(failed_build) > 0:
            print(
                f"WARNING: EXCLUDING {len(failed_build)} DATAPOINTS BECAUSE BUILDING DOCKER CONTAINERS FOR THEM FAILED."
            )

        self.containers = threaded_map(
            (
                delayed(build_container)(
                    test_spec=test_spec,
                    client=self.docker_client,
                    run_id="run",
                    logger=logging.getLogger(__name__),
                    nocache=False,
                    force_rebuild=False,
                )
                for test_spec in successfully_built
            ),
            max_workers=self.max_workers,
            tqdm_description="building containers",
        )  # type: ignore

        instance_id_to_datapoint: dict[str, dict] = {
            datapoint["instance_id"]: datapoint  # type: ignore
            for datapoint in self.dataset
        }
        self.dataset = [
            instance_id_to_datapoint[
                test_spec.instance_id.removesuffix(unique_identifier)
            ]
            for test_spec in successfully_built
        ]

    def _start_containers(self) -> None:
        threaded_map(
            (delayed(container.start)() for container in self.containers),
            max_workers=self.max_workers,
            tqdm_description="starting containers",
        )

    def _stop_containers(self) -> None:
        threaded_map(
            (delayed(container.stop)() for container in self.containers),
            max_workers=self.max_workers,
            tqdm_description="stopping containers",
        )

    def __enter__(self) -> "SweBenchEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_containers()

    def solution_evaluation_scores(self) -> list[float]:
        return threaded_map(
            (
                delayed(swe_bench_solution_evaluation_score)(container, datapoint)
                for container, datapoint in list(
                    zip(self.containers, self.dataset, strict=True)
                )
            ),
            max_workers=self.max_workers,
            tqdm_description="evaluating solutions",
        )  # type: ignore

    def _filter_dataset(self, instance_ids: list[str]) -> None:
        assert pairwise_distinct(instance_ids)
        assert set(instance_ids) <= set(
            datapoint["instance_id"]  # type: ignore
            for datapoint in self.dataset
        )

        instance_id_to_datapoint = {
            datapoint["instance_id"]: datapoint  # type: ignore
            for datapoint in self.dataset
        }
        self.dataset = [
            instance_id_to_datapoint[instance_id] for instance_id in instance_ids
        ]


@beartype
def swe_bench_solution_evaluation_score(container: Container, datapoint: dict) -> float:
    # stolen from UK AISI's Inspect library without their consent
    # TO DO: check if the license allows doing this

    container.exec_run(CREATE_MODEL_PATCH.format(base_commit=datapoint["base_commit"]))

    eval_script: str = get_eval_script(
        test_patch=datapoint["test_patch"],
        repo=datapoint["repo"],
        version=datapoint["version"],
        base_commit=datapoint["base_commit"],
    )

    upload_file(container, filename="eval_script", content=eval_script)

    container.exec_run("chmod +x ./eval_script")
    eval_script_output = container.exec_run("./eval_script").output

    score, score_explanation = get_score_and_explanation_from_test_output(
        test_output=eval_script_output.decode("utf-8"),
        state=TaskState(
            model="",  # type: ignore
            sample_id="",
            epoch=0,
            input="",
            messages=[],
            metadata={
                key: (
                    json.loads(value)
                    if key in ["FAIL_TO_PASS", "PASS_TO_PASS"]
                    else value
                )
                for key, value in datapoint.items()
            },
        ),
    )

    return score


CREATE_MODEL_PATCH = """cd /testbed
git add -A
git diff --cached {base_commit} > model.patch"""


@beartype
def upload_file(container: Container, *, filename: str, content: str) -> None:
    # use base64 as a cursed way to escape
    base64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    output = container.exec_run(
        ["/bin/bash", "-c", f"echo {base64_content} | base64 -d > '{filename}'"]
    )
    assert output.exit_code == 0


@beartype
@dataclass
class BashTool(Tool):
    """
    Wrapper arond a docker `Container` that allows an LLM agent to execute bash commands in the container and see their outputs.
    """

    container: Container
    max_output_length: int = 1024
    timeout_seconds: int = 60

    def name(self) -> str:
        return "bash"

    def description(self) -> str:
        return "Execute a bash command."

    def argument_description(self) -> dict[str, str] | str | None:
        return "The bash command."

    def _call(self, command: str) -> str:
        execution_result = self.container.exec_run(
            ["timeout", str(self.timeout_seconds), "/bin/bash", "-c", command]
        )
        output = self._truncate(
            execution_result.output.decode("utf-8", errors="replace")
        )
        return (
            f"<exit_code>{execution_result.exit_code}</exit_code>\n"
            f"<output>\n{output}\n</output>\n"
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_length:
            return text
        return (
            text[: self.max_output_length // 2]
            + "[TRUNCATED]"
            + text[-self.max_output_length // 2 :]
        )


@beartype
def pairwise_distinct(xs: Iterable) -> bool:
    if not isinstance(xs, list):
        xs = list(xs)
    return len(xs) == len(set(xs))
