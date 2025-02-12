from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from itertools import chain
from collections.abc import Callable, Iterable
from typing import Any
from beartype import beartype


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


@beartype
def nested_threaded_map(
    delayed_functions: Iterable[Iterable[Callable[[], Any]]],
    max_workers: int,
    tqdm_description: str | None = None,
) -> list[list[Any]]:
    """
    Use `threaded_map([[delayed(f)(...) for ... in ...] for ... in ...])` to run `[[f(...) for ... in ...] for ... in ...]` in a threaded way.
    """

    delayed_functions = [list(fs) for fs in delayed_functions]
    flattened_delayed_functions = list(chain.from_iterable(delayed_functions))
    flattened_results = threaded_map(
        flattened_delayed_functions,
        max_workers=max_workers,
        tqdm_description=tqdm_description,
    )

    results: list[list[Any]] = []
    i = 0
    for fs in delayed_functions:
        results.append(flattened_results[i : i + len(fs)])
        i += len(fs)
    assert i == len(flattened_results)
    return results
