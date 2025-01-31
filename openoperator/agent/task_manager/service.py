import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    Represents a single task with optional key-value pairs.
    """

    description: str
    additional_information: Optional[Dict[str, str]] = None


class TaskManager:
    """
    A separate class responsible for managing the Agent's tasks.
    It handles all logic related to adding, removing, and
    iterating through tasks.
    """

    def __init__(self):
        # Holds the list of Task objects
        self.tasks: List[Task] = []
        # Tracks which task is currently being executed
        self.current_task_index: int = 0

    def has_tasks(self) -> bool:
        """Return True if there are tasks to process."""
        return len(self.tasks) > 0

    def is_all_done(self) -> bool:
        """Return True if we've processed all tasks."""
        return self.current_task_index >= len(self.tasks)

    def get_current_task(self) -> Optional[Task]:
        """
        Returns the current task to be executed.
        If there is no current task (out of range), returns None.
        """
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def increment_task_index(self) -> None:
        """
        Move on to the next task in the queue.
        Note: Caller is responsible for ensuring we don't go out of range.
        """
        self.current_task_index += 1
        logger.debug(f"Incremented task index to {self.current_task_index}.")

    def add_task(
        self, description: str, additional_information: Optional[Dict[str, str]] = None, index: Optional[int] = None
    ) -> None:
        """
        Add a single new task to the tasks list.

        - `description`: Description of the task.
        - `additional_information`: Optional key-value pairs associated with the task. Useful to pass credentials, file paths, etc.
        - If `index` is None, the task is appended to the end.
        - If an `index` is specified, the task is inserted at that position.
        - You cannot insert a task before the currently running (or completed) task.
        """

        new_task = Task(description=description, additional_information=additional_information)

        if index is None:
            self.tasks.append(new_task)
        else:
            # Ensure we're not inserting into the past
            if index < self.current_task_index:
                raise ValueError(
                    f"Cannot insert a new task at index={index}, which is before the current or already completed tasks."
                )
            self.tasks.insert(index, new_task)

    def remove_task(self, index: Optional[int] = None) -> None:
        """
        Remove a single pending task from the tasks list by index.

        - If no index is provided, remove the last pending task.
        - You can only remove tasks that haven't started (index >= current_task_index).
        """
        if not self.tasks:
            logger.warning('No tasks to remove.')
            return

        if index is None:
            # Remove the last pending task, if any
            last_index = len(self.tasks) - 1
            if last_index < self.current_task_index:
                raise ValueError('No pending tasks left to remove (all tasks are completed or in-progress).')
            self.tasks.pop()
        else:
            if index < self.current_task_index or index >= len(self.tasks):
                raise ValueError('Index out of range or task is already completed/in progress.')
            self.tasks.pop(index)

    def list_tasks(self) -> List[Task]:
        """
        Returns a list of all tasks.
        """
        return self.tasks

    def clear_all_tasks(self) -> None:
        """
        Clears all tasks from the manager.
        """
        self.tasks.clear()
        self.current_task_index = 0
        logger.info("Cleared all tasks from the TaskManager.")
