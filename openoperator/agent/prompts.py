from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from openoperator.agent.task_manager.service import Task
from openoperator.agent.views import ActionResult
from openoperator.browser.views import BrowserState


class SystemPrompt:
    """
    Creates the system prompt for the main agent logic.
    """

    def __init__(
        self,
        action_description: str,
        current_date: datetime,
        max_actions_per_step: int = 10,
    ):
        self.default_action_description = action_description
        self.current_date = current_date
        self.max_actions_per_step = max_actions_per_step

    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = f"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {{
     "current_state": {{
       "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not",
       "memory": "Description of what has been done and what you need to remember until the end of the task",
       "next_goal": "What needs to be done with the next actions"
     }},
     "action": [
       {{
         "one_action_name": {{
           // action-specific parameter
         }}
       }},
       // ... more actions in sequence
     ]
   }}

2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item.

   Common action sequences:
   - Form filling: [
       {{"input_text": {{"index": 1, "text": "username"}}}},
       {{"input_text": {{"index": 2, "text": "password"}}}},
       {{"click_element": {{"index": 3}}}}
     ]
   - Navigation and extraction: [
       {{"open_new_tab": {{}}}},
       {{"go_to_url": {{"url": "https://example.com"}}}},
       {{"extract_page_content": {{}}}}
     ]


3. ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list
   - Each element has a unique index number (e.g., "33[:]<button>")
   - Elements marked with "_[:]" are non-interactive (for context only)

4. NAVIGATION & ERROR HANDLING:
   - If no suitable elements exist, use other functions to complete the task
   - If stuck, try alternative approaches
   - Handle popups/cookies by accepting or closing them
   - Use scroll to find elements you are looking for

5. TASK COMPLETION:
   - Use the done action as the last action as soon as the task is complete
   - Don't hallucinate actions
   - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
   - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.

6. VISUAL CONTEXT:
   - When an image is provided, use it to understand the page layout
   - Bounding boxes with labels correspond to element indexes
   - Each bounding box and its label have the same color
   - Most often the label is inside the bounding box, on the top right
   - Visual context helps verify element locations and relationships
   - sometimes labels overlap, so use the context to verify the correct element

7. Form filling:
   - If you fill an input field and your action sequence is interrupted, most often a list with suggestions popped up under the field and you need to first select the right element from the suggestion list.

8. ACTION SEQUENCING:
   - Actions are executed in the order they appear in the list
   - Each action should logically follow from the previous one
   - If the page changes after an action, the sequence is interrupted and you get the new state.
   - If content only disappears the sequence continues.
   - Only provide the action sequence until you think the page will change.
   - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
   - only use multiple actions if it makes sense.

   - use maximum {self.max_actions_per_step} actions per sequence
"""
        return text

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Current URL: The webpage you're currently on
2. Available Tabs: List of open browser tabs
3. Interactive Elements: List in the format:
   index[:]<element_type>element_text</element_type>
   - index: Numeric identifier for interaction
   - element_type: HTML element type (button, input, etc.)
   - element_text: Visible text or element description

Example:
33[:]<button>Submit Form</button>
_[:] Non-interactive text

Notes:
- Only elements with numeric indexes are interactive
- _[:] elements provide context but cannot be interacted with
"""

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.
        """
        time_str = self.current_date.strftime('%Y-%m-%d %H:%M')
        AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
1. Analyze the provided webpage elements and structure
2. Plan a sequence of actions to accomplish the given task
3. Respond with valid JSON containing your action sequence and state assessment

Current date and time: {time_str}

{self.input_format()}

{self.important_rules()}

Functions:
{self.default_action_description}

Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid.
"""
        return SystemMessage(content=AGENT_PROMPT)


class AgentMessagePrompt:
    """
    Formats the user-facing (human) message that includes:
      - current URL
      - list of tabs
      - a representation of the DOM / clickable elements
      - results from the last action
      - (optionally) a screenshot in base64
    """

    def __init__(
        self,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        use_vision: bool = False,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
    ):
        self.state = state
        self.result = result
        self.use_vision = use_vision
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes

    def get_user_message(self) -> HumanMessage:
        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)
        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0

        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'

        state_description = f"""
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements from current page view:
{elements_text}
"""

        if self.result:
            for i, action_result in enumerate(self.result):
                if action_result.extracted_content:
                    state_description += f'\nAction result {i + 1}/{len(self.result)}: {action_result.extracted_content}'
                if action_result.error:
                    # only use last max_error_length characters of error
                    error_snippet = action_result.error[-self.max_error_length :]
                    state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error_snippet}'

        # If there's a screenshot, format for a vision model
        if self.state.screenshot and self.use_vision:
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )

        # Otherwise return plain text
        return HumanMessage(content=state_description)


class ValidatorSystemPrompt:
    """
    Prompt used during the optional validation step to confirm if the agent’s result
    is what the user intended (e.g., if the output is correct, relevant, or needs more actions).
    """

    def __init__(self, task: str):
        self.task = task

    def get_system_message(self) -> SystemMessage:
        text = (
            'You are a validator of an agent who interacts with a browser. '
            'Validate if the output of the last action is what the user wanted '
            'and if the task is completed. If the task is unclear, you can let it pass. '
            'If something is missing or the image does not show what was requested, do not let it pass. '
            'Try to understand the page and help the model with suggestions like scroll, click, etc. '
            'Return a JSON object with 2 keys: is_valid (bool) and reason (string). '
            f'Task to validate: "{self.task}". '
            'Example: {"is_valid": false, "reason": "The user wanted to search for cat photos, but the agent searched for dog photos."}'
        )
        return SystemMessage(content=text)


class TaskPrompt:

    def __init__(self, task: Task):
        self.task = task

    def get_user_message(self) -> HumanMessage:
        content = (
            f'Your ultimate task is: {self.task.description}. '
            'If you achieved your ultimate task, stop everything and use the done action in the next step to complete '
            'the task. If not, continue as usual.'
        )

        if self.task.additional_information:

            placeholder_string = ", ".join([f"***{key}***" for key in self.task.additional_information])

            content += f'You can use {placeholder_string} as placeholders in actions that allow it. The placeholders will be automatically be replaced.'

        return HumanMessage(content=content)
