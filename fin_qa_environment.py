# Standard imports
import json

# Third Party Imports
from rllm.environments.tools.tool_env import ToolEnvironment

# Local imports
from .fin_qa_reward_doubao import fin_qa_reward_function
from .fin_qa_tools import Calculator, GetTableInfo, GetTableNames, SQLQuery


class FinQAEnvironment(ToolEnvironment):
    """
    The Financial question answering environment with tool support.
    """

    def __init__(self, task: dict | None = None):
        """
        Initialize the FinQAEnvironment.

        Args:
            task: The path to the FinQA task.
        """
        # tool map for FinQA
        tool_map = {
            "calculator": Calculator,
            "get_table_info": GetTableInfo,
            "get_table_names": GetTableNames,
            "sql_query": SQLQuery,
        }

        super().__init__(task=task, tool_map=tool_map, reward_fn=fin_qa_reward_function, max_steps=20)
        self.accessed_tables = []  # track accessed tables in get_table_info tool call

    def reset(self, task: dict | None = None):
        """Reset environment and clear accessed tables/task"""
        if task is not None:
            self.task = task

        # Clear state from task if it exists
        if hasattr(self, "task") and self.task is not None:
            self.task.pop("accessed_tables", None)
            self.task.pop("tool_stats", None)
            self.task.pop("num_tool_calls", None)

        self.accessed_tables = []
        self.tool_stats = {}
        self.num_tool_calls = 0

        return super().reset()

    def _execute_tool_calls(self, tool_calls: list[dict]) -> dict[str, str]:
        """Track accessed tables and tool usage statistics"""
        tool_outputs = super()._execute_tool_calls(tool_calls)
        
        self.num_tool_calls += len(tool_calls)
        
        for tool_call in tool_calls:
            # 1. Track accessed tables
            if tool_call.get("function", {}).get("name") == "get_table_info":
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    table_name = tool_args.get("table_name", "")
                    if table_name:
                        self.accessed_tables.append(table_name.lower().strip())
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # 2. Track Tool Success/Fail 
            tool_name = tool_call.get("function", {}).get("name", "unknown")
            output_str = tool_outputs.get(tool_call.get("id"), "")
            
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = {"success": 0, "fail": 0}
                
            if "Error:" in output_str or "Error :" in output_str or "Error evaluating expression" in output_str:
                self.tool_stats[tool_name]["fail"] += 1
            else:
                self.tool_stats[tool_name]["success"] += 1

        if self.task is not None:
            self.task["accessed_tables"] = self.accessed_tables
            self.task["tool_stats"] = self.tool_stats
            self.task["num_tool_calls"] = self.num_tool_calls

        return tool_outputs

    @staticmethod
    def from_dict(env_args: dict) -> "FinQAEnvironment":
        if "task" in env_args:
            task = env_args["task"]
        else:
            task = env_args
        return FinQAEnvironment(task=task)
