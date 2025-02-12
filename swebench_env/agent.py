from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
import re
from itertools import chain
from abc import ABC, abstractmethod
from dataclasses import dataclass
from beartype import beartype

from .threaded_map import nested_threaded_map, delayed


@beartype
class Tool(ABC):
    """
    Base class for a tool than an LLM agent can use.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the tool.
        To use the tool, the agent will have to write <name>arguments</name>, where name is the string this function returns.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        Returns the description of the tool.
        It will be included in the LLM agent's prompt.
        """
        pass

    @abstractmethod
    def argument_description(self) -> dict[str, str] | str | None:
        """
        Returns the descriptions of the arguments given to the tool in English.
        They will be included in the LLM agent's prompt.
        If the tool does not take arguments, returns `None`.
        If the tool takes one argument, returns a string, which is the description of the argument in English.
        If the tools takes multiple arguments, returns a dictionary mapping the names of the arguments to their descriptions in English.
        """
        pass

    @abstractmethod
    def _call(self, *args, **kwargs) -> str:
        """
        Override this method to execute the tool call.
        It should take no arguments, one positional argument, or keyword arguments if `self.argument_description()` is `None`, a `str`, or a `dict[str, str]`, respectively. In the latter case, the namse of the keyword arguments it takes should be the same as `self.argument_description().keys()`.
        It should return the output of the tool as a string.
        If the tool does not have an output, return something like "Tool call successful." or "".
        """
        pass

    def __post_init__(self) -> None:
        assert self.name() != "think", (
            "A tool cannot be named 'think' because this would conflict with DeepSeek-R1's <think>...</think> chain of thought tags."
        )

    def call_arguments_valid(self, arguments: dict[str, str] | str | None) -> bool:
        argument_description = self.argument_description()

        if argument_description is None:
            return arguments is None

        if isinstance(argument_description, str):
            return isinstance(arguments, str)

        if isinstance(argument_description, dict):
            return isinstance(arguments, dict) and set(arguments.keys()) == set(
                argument_description.keys()
            )

        assert False, (
            "Tool.argument_description should return an object of type dict[str, str] | str | None."
        )

    def __call__(self, arguments: dict[str, str] | str | None) -> str:
        if not self.call_arguments_valid(arguments):
            return f"An invalid combination of arguments was provided to the tool {self.name()}."

        try:
            if arguments is None:
                return self._call()
            if isinstance(arguments, str):
                return self._call(arguments)
            if isinstance(arguments, dict):
                return self._call(*arguments)
            return "Tool call failed."
        except Exception:
            return "Tool call failed."

    def usage_message(self) -> str:
        message = f"# Tool: {self.name()}\n"
        message += "\n"
        message += self.description() + "\n"
        message += "\n"
        message += "To use the tool, simply write the following:\n"
        message += "\n"

        argument_description = self.argument_description()

        if argument_description is None:
            message += f"<{self.name()}/>"
            return message

        if isinstance(argument_description, str):
            message += f"<{self.name()}>\n"
            message += argument_description + "\n"
            message += f"</{self.name()}>"
            return message

        if isinstance(argument_description, dict):
            message += f"<{self.name()}>\n"
            for argument_name, description in argument_description.items():
                message += f"<{argument_name}>{description}</{argument_name}>\n"
            message += f"</{self.name()}>"

        assert False, (
            "Tool.argument_description should return an object of type dict[str, str] | str | None."
        )


@beartype
def run_agent(
    llm: LLM,
    prompts: list[str],
    tools: list[list[Tool]],
    max_turns: int = 16,
    can_finish: bool = True,
    sampling_params: SamplingParams = SamplingParams(temperature=0, max_tokens=1024),
    verbose: bool = False,
    tool_max_workers: int = 1,
) -> None:
    """
    Runs `len(prompts)` LLM agent loops in parallel.

    Arguments:
    :llm: The LLM. For better performance, pass `enable_prefix_caching=True` when initializing it.
    :prompts: List of prompts. One agent will be ran with each prompt. A prompt explaining to the LLM that it should act as an agent and how to use the tools will be added to those prompts.
    :tools: Tools that each agent can use. The agent with prompt `prompts[i]` can use the tools `tools[i]`.
    :max_turns: maximal number of messages the LLM can send per agent loop.
    :can_finish: Whether the agents can finish the agent loop before having sent `max_turn` messages.
    :sampling_params: VLLM sampling parameters for calling the LLM.
    :verbose: Print a transcripts of all the LLM prompts and responses when the agent loops are finished running.
    """

    assert len(tools) == len(prompts)

    assert not any(
        isinstance(tool, FinishTool) for tool in chain.from_iterable(tools)
    ), (
        "Do not include FinishTool in the list of tools. It will be added automatically if necessary."
    )

    if can_finish:
        for t in tools:
            t.append(FinishTool())

    conversations: list[list[ChatCompletionMessageParam]] = [
        [
            {
                "role": "user",
                "content": agent_instruction_message(
                    prompt=prompt,
                    tools=tools_,
                    can_finish=can_finish,
                ),
            }
        ]
        for prompt, tools_ in zip(prompts, tools, strict=True)
    ]

    unfinished_conversation_indices: list[int] = list(range(len(conversations)))

    for _ in range(max_turns):
        unfinished_conversations = [
            conversations[i] for i in unfinished_conversation_indices
        ]
        completions = llm.chat(
            unfinished_conversations, sampling_params=sampling_params
        )
        completions: list[str] = [
            completion.outputs[0].text for completion in completions
        ]

        tool_calls: list[list[ToolCall]] = [
            extract_tool_calls(completion) for completion in completions
        ]

        tool_call_results: list[list[ToolCallResult]] = nested_threaded_map(
            [
                [delayed(call_tool)(call, all_tools=tools[i]) for call in calls]
                for i, calls in zip(
                    unfinished_conversation_indices, tool_calls, strict=True
                )
            ],
            max_workers=tool_max_workers,
            tqdm_description="executing tool calls",
        )

        for i_conversation, completion, tool_results in zip(
            unfinished_conversation_indices.copy(),  # this .copy() is important because we delete elements from unfinished_conversation_indices inside this loop
            completions,
            tool_call_results,
            strict=True,
        ):
            conversations[i_conversation].append(
                {"role": "assistant", "content": completion}
            )

            if len(tool_calls) == 0:
                conversations[i_conversation].append(
                    {
                        "role": "user",
                        "content": "Please call at least one tool in each message.",
                    }
                )

            for tool_call_result in tool_results:
                if (
                    tool_call_result.finish_tool_called
                    and i_conversation in unfinished_conversation_indices
                ):
                    unfinished_conversation_indices.remove(i_conversation)

                conversations[i_conversation].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_result.tool_name,
                        "content": tool_call_result.tool_response,
                    }
                )

    if verbose:
        print("+" * 100)
        print("+" * 100)
        print("+" * 100)
        print("FINISHED EXECUTING AGENT LOOPS")
        for i, conversation in enumerate(conversations):
            print("+" * 100)
            print("+" * 100)
            print("+" * 100)
            print(f"AGENT LOOP {i + 1}")
            for message in conversation:
                print("=" * 100)
                for field, value in message.items():
                    if field == "content":
                        continue
                    print(field.upper(), value)
                print("CONTENT:", message["content"])


@beartype
def agent_instruction_message(prompt: str, tools: list[Tool], can_finish: bool) -> str:
    message = "You are an agent which has access to tools. You have a goal which you should accomplish. In order to call tools, write tool calls, formatted exactly as in the examples. You will see the outputs of those tool calls in the next user message. If you want to do multiple things in sequence, you must do them one by one, returning control to the user after each thing, as you will only see the output of tool calls to do one thing in the next user message. Be thorough and do not give up - if you did something and it failed, you should fix it or do something eles that will achieve your goal that doesn't fail. When you think you are done with your goal, you should test whether you are actually done and everything you did actually works. If it doesn't, you should either fix it or start over with a different approach.\n"
    if can_finish:
        message += "When you are done, you should make sure that you have indeed accomplished the goal before calling finish_tool. If you realize that you have not accomplished it, you should try again and not call finish_tool until you are sure you accomplished your goal.\n"
    message += "\n"
    message += "You can use the following tools. To use a tool, you must format the tool call exactly as it is formatted below, or the tool call will not work.\n"
    message += "\n"
    message += "\n\n".join(tool.usage_message() for tool in tools)
    message += "\n"
    message += "\n"
    message += "# Your Goal\n"
    message += "\n"
    message += prompt

    return message


@beartype
@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, str] | str | None


@beartype
@dataclass
class ToolCallResult:
    finish_tool_called: bool
    tool_name: str
    tool_response: str


@beartype
def call_tool(tool_call: ToolCall, all_tools: list[Tool]) -> ToolCallResult:
    tool = find_tool_by_name(tool_call.tool_name, all_tools)

    if tool is None:
        return ToolCallResult(
            finish_tool_called=False,
            tool_name=tool_call.tool_name,
            tool_response=f"Error: There is no tool named '{tool_call.tool_name}'.",
        )

    if isinstance(tool, FinishTool):
        return ToolCallResult(
            finish_tool_called=True,
            tool_name=tool_call.tool_name,
            tool_response="Finished.",
        )

    tool_response = tool(tool_call.arguments)

    return ToolCallResult(
        finish_tool_called=False,
        tool_name=tool_call.tool_name,
        tool_response=tool_response,
    )


@beartype
def find_tool_by_name(tool_name: str, tools: list[Tool]) -> Tool | None:
    matching_tools = [tool for tool in tools if tool.name() == tool_name]

    if len(matching_tools) == 0:
        return None

    if len(matching_tools) > 1:
        print(
            f"WARNING: Found more than one tool with name '{tool_name}'. Tool names should be unique."
        )

    return matching_tools[0]


@beartype
@dataclass
class FinishTool(Tool):
    """
    Special tool to be used by the agent if it finished its task.
    It is treated specially by the agent loop.
    """

    def name(self) -> str:
        return "finish"

    def description(self) -> str:
        return "Call this tool when you are finished. Only call it when you are very sure that you indeed finished your task correctly, and made the best effort possible to verify that what you did is indeed correct."

    def argument_description(self) -> dict[str, str] | str | None:
        return None

    def _call(self) -> str:
        assert False, "Do not call FinishTool.__call__."


@beartype
@beartype
def extract_tool_calls(llm_message: str) -> list[ToolCall]:
    """
    Extracts tool calls of the following forms:
      1) <tool_name/>               (no arguments)
      2) <tool_name>single_argument</tool_name>   (string argument)
      3) <tool_name><arg>val</arg>...</tool_name> (dictionary arguments)
      4) ```tool_name
         single_argument
         ```                (backtick format)
         (Similarly, can also hold <arg>val</arg> blocks interpreted as a dict.)

    Returns a list of ToolCall objects.
    """

    # We define a single combined pattern with alternation to capture:
    #   * The triple-backtick form: ```tool_name ... ```
    #   * The self-closing XML form: <tool_name/>
    #   * The open/close XML form: <tool_name>...</tool_name>
    #
    # Explanation of capture groups:
    #   (1) triple_name, (2) triple_content
    #   (3) self_closing_name
    #   (4) open_close_name, (5) inner_content
    pattern = re.compile(
        r"""
        # 1) The triple-backtick form: ```tool_name ... ```
        ```(\w+)\s+(.*?)```

        |

        # 2) A self-closing tag: <tool_name/>
        <(\w+)\s*/>

        |

        # 3) An open/close pair: <tool_name>...</tool_name>
        <(\w+)>(.*?)</\4>
        """,
        re.DOTALL | re.VERBOSE,
    )

    # Pattern used to detect dictionaries of the form <key>value</key>
    # repeated multiple times.
    dict_pattern = re.compile(r"^(?:\s*<(\w+)>(.*?)</\1>\s*)+$", re.DOTALL)

    matches = pattern.findall(llm_message)
    tool_calls: list[ToolCall] = []

    for (
        triple_name,
        triple_content,
        self_closing_name,
        open_close_name,
        inner_content,
    ) in matches:
        if triple_name:
            # Matched the triple-backtick pattern: ```tool_name ... ```
            tool_name = triple_name
            content_stripped = triple_content.strip()

            # Check if content is a series of <arg>value</arg> blocks
            if dict_pattern.fullmatch(content_stripped):
                pairs = re.findall(r"<(\w+)>(.*?)</\1>", content_stripped, re.DOTALL)
                arguments_dict = {k: v.strip() for k, v in pairs}
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=arguments_dict)
                )
            else:
                # Otherwise treat as a single string argument
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=content_stripped)
                )

        elif self_closing_name:
            # Matched the self-closing pattern <tool_name/>
            tool_calls.append(ToolCall(tool_name=self_closing_name, arguments=None))

        else:
            # Matched the open/close pattern <tool_name>...</tool_name>
            tool_name = open_close_name
            content_stripped = inner_content.strip()

            # Check if content is purely made up of <arg>value</arg> blocks
            if dict_pattern.fullmatch(content_stripped):
                # Parse into a dictionary
                pairs = re.findall(r"<(\w+)>(.*?)</\1>", content_stripped, re.DOTALL)
                arguments_dict = {k: v.strip() for k, v in pairs}
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=arguments_dict)
                )
            else:
                # Otherwise treat as a single string argument
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=content_stripped)
                )

    # Filter out tool calls whose name is "think"
    # (Some systems reserve <think>...</think> as scratchpad, etc.)
    tool_calls = [tc for tc in tool_calls if tc.tool_name != "think"]

    return tool_calls


r"""
@beartype
@beartype
def extract_tool_calls(llm_message: str) -> list[ToolCall]:
    " " "
    Extracts tool calls of the following forms:
      1) <tool_name/>  (no arguments)
      2) <tool_name>single_argument</tool_name>  (string argument)
      3) <tool_name><arg>val</arg>...</tool_name>  (dictionary arguments)

    Returns a list of ToolCall objects.
    " " "

    # Regex to match either:
    #   * A self-closing tag (<tool_name/>), or
    #   * An open/close pair (<tool_name> ... </tool_name>)
    #     capturing the tool_name and everything inside.
    pattern = re.compile(
        r" " "
        # 1) A self-closing tag: <tool_name/>
        <(\w+)\s*/>

        |

        # 2) An open/close pair: <tool_name>...</tool_name>
        <(\w+)>(.*?)</\2>
        " " ",
        re.DOTALL | re.VERBOSE,
    )

    matches = pattern.findall(llm_message)
    tool_calls: list[ToolCall] = []

    for self_closing_name, open_close_name, inner_content in matches:
        if self_closing_name:
            # Matched the self-closing pattern <tool_name/>
            tool_calls.append(ToolCall(tool_name=self_closing_name, arguments=None))
        else:
            # Matched the open/close pattern <tool_name> ... </tool_name>
            tool_name = open_close_name
            content_stripped = inner_content.strip()

            # Check if content is purely made up of <arg>value</arg> blocks
            # Possibly multiple lines, so we use a "fullmatch" that allows
            # repeated <key>value</key> segments with optional whitespace.
            dict_pattern = re.compile(r"^(?:\s*<(\w+)>(.*?)</\1>\s*)+$", re.DOTALL)

            if dict_pattern.fullmatch(content_stripped):
                # Parse into a dictionary
                # Extract all <key>value</key> pairs
                pairs = re.findall(r"<(\w+)>(.*?)</\1>", content_stripped, re.DOTALL)
                arguments_dict = {k: v.strip() for k, v in pairs}
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=arguments_dict)
                )
            else:
                # Otherwise treat as a single string argument
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=content_stripped)
                )

    # don't include tool calls with name thing because deepseek uses <think>...</think> tags for the scratchpad
    tool_calls = [
        tool_call for tool_call in tool_calls if tool_call.tool_name != "think"
    ]

    return tool_calls
"""
