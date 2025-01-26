from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import platform
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from lmnr import observe
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from openoperator.agent.message_manager.service import MessageManager
from openoperator.agent.prompts import (
    AgentMessagePrompt,
    SystemPrompt,
    ValidatorSystemPrompt,
)
from openoperator.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentStepInfo,
)
from openoperator.browser.browser import Browser
from openoperator.browser.context import BrowserContext
from openoperator.browser.views import BrowserState, BrowserStateHistory
from openoperator.controller.registry.views import ActionModel
from openoperator.controller.service import Controller
from openoperator.dom.history_tree_processor.service import (
    DOMHistoryElement,
    HistoryTreeProcessor,
)
from openoperator.telemetry.service import ProductTelemetry
from openoperator.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from openoperator.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class Agent:
    def __init__(
        self,
        task: Union[str, List[str], None],  # Accept single task, a list of tasks of no task at all for late initialization
        llm: BaseChatModel,
        browser: Browser | None = None,
        browser_context: BrowserContext | None = None,
        controller: Controller = Controller(),
        use_vision: bool = True,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        max_failures: int = 3,
        retry_delay: int = 10,
        system_prompt_class: Type[SystemPrompt] = SystemPrompt,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        message_context: Optional[str] = None,
        generate_gif: bool | str = True,
        include_attributes: list[str] = [
            'title',
            'type',
            'name',
            'role',
            'tabindex',
            'aria-label',
            'placeholder',
            'value',
            'alt',
            'aria-expanded',
        ],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        tool_call_in_content: bool = True,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        # Cloud Callbacks
        register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], None] | None = None,
        register_done_callback: Callable[['AgentHistoryList'], None] | None = None,
        tool_calling_method: Optional[str] = 'auto',
        reset_messages_on_new_task: bool = False,
    ):
        self.agent_id = str(uuid.uuid4())  # unique identifier for the agent

        # Normalize tasks into a list internally
        if task is not None:
            if isinstance(task, str):
                self.tasks = [task]
            else:
                self.tasks = task

        # We'll keep track of the "current" task in self.task
        # so that existing references (prompts, logs) still work.
        # When running multiple tasks, we update this as we go.
        self.task = self.tasks[0] if self.tasks else None

        self.use_vision = use_vision
        self.llm = llm
        self.save_conversation_path = save_conversation_path
        self.save_conversation_path_encoding = save_conversation_path_encoding
        self._last_result = None
        self.include_attributes = include_attributes
        self.max_error_length = max_error_length
        self.generate_gif = generate_gif

        # Controller setup
        self.controller = controller
        self.max_actions_per_step = max_actions_per_step

        # Browser setup
        self.injected_browser = browser is not None
        self.injected_browser_context = browser_context is not None
        self.message_context = message_context

        # Initialize browser first if needed
        self.browser = browser if browser is not None else (None if browser_context else Browser())

        # Initialize browser context
        if browser_context:
            self.browser_context = browser_context
        elif self.browser:
            self.browser_context = BrowserContext(browser=self.browser, config=self.browser.config.new_context_config)
        else:
            # If neither is provided, create both new
            self.browser = Browser()
            self.browser_context = BrowserContext(browser=self.browser)

        self.system_prompt_class = system_prompt_class

        # Telemetry setup
        self.telemetry = ProductTelemetry()

        # Action and output models setup
        self._setup_action_models()
        self._set_version_and_source()
        self.max_input_tokens = max_input_tokens

        self._set_model_names()

        self.tool_calling_method = self._set_tool_calling_method(tool_calling_method)

        self.message_manager = MessageManager(
            llm=self.llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            message_context=self.message_context,
        )

        # Step callback
        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback

        # Tracking variables
        self.history: AgentHistoryList = AgentHistoryList(history=[])
        self.n_steps = 1
        self.consecutive_failures = 0
        self.max_failures = max_failures
        self.retry_delay = retry_delay
        self.validate_output = validate_output
        self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None
        if save_conversation_path:
            logger.info(f'Saving conversation to {save_conversation_path}')

        self._paused = False
        self._stopped = False

        self.reset_messages_on_new_task = reset_messages_on_new_task

    def _set_version_and_source(self) -> None:
        try:
            import pkg_resources

            version = pkg_resources.get_distribution('openoperator').version
            source = 'pip'
        except Exception:
            try:
                import subprocess

                version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
                source = 'git'
            except Exception:
                version = 'unknown'
                source = 'unknown'
        logger.debug(f'Version: {version}, Source: {source}')
        self.version = version
        self.source = source

    def _set_model_names(self) -> None:
        self.chat_model_library = self.llm.__class__.__name__
        if hasattr(self.llm, 'model_name'):
            self.model_name = self.llm.model_name  # type: ignore
        elif hasattr(self.llm, 'model'):
            self.model_name = self.llm.model  # type: ignore
        else:
            self.model_name = 'Unknown'

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

    def _set_tool_calling_method(self, tool_calling_method: Optional[str]) -> Optional[str]:
        if tool_calling_method == 'auto':
            if self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None

    @time_execution_async('--step')
    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        logger.info(f'\n📍 Step {self.n_steps}')
        state = None
        model_output = None
        result: list[ActionResult] = []

        try:
            state = await self.browser_context.get_state(use_vision=self.use_vision)

            if self._stopped or self._paused:
                logger.debug('Agent paused after getting state')
                raise InterruptedError

            self.message_manager.add_state_message(state, self._last_result, step_info)
            input_messages = self.message_manager.get_messages()

            try:
                model_output = await self.get_next_action(input_messages)

                if self.register_new_step_callback:
                    self.register_new_step_callback(state, model_output, self.n_steps)

                self._save_conversation(input_messages, model_output)
                self.message_manager._remove_last_state_message()  # remove large state chunk from chat

                if self._stopped or self._paused:
                    logger.debug('Agent paused after getting next action')
                    raise InterruptedError

                self.message_manager.add_model_output(model_output)
            except Exception as e:
                # model call failed, remove last state message from history
                self.message_manager._remove_last_state_message()
                raise e

            result: list[ActionResult] = await self.controller.multi_act(model_output.action, self.browser_context)
            self._last_result = result

            # TODO add call to verify if goal has been achieved

            if len(result) > 0 and result[-1].is_done:
                logger.info(f'📄 Result: {result[-1].extracted_content}')

            self.consecutive_failures = 0

        except InterruptedError:
            logger.debug('Agent paused')
            return
        except Exception as e:
            result = await self._handle_step_error(e)
            self._last_result = result

        finally:
            actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.agent_id,
                    step=self.n_steps,
                    actions=actions,
                    consecutive_failures=self.consecutive_failures,
                    step_error=([r.error for r in result if r.error] if result else ['No result']),
                )
            )
            if not result:
                return

            if state:
                self._make_history_item(model_output, state, result)

    async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
        """Handle all types of errors that can occur during a step"""
        include_trace = logger.isEnabledFor(logging.DEBUG)
        error_msg = AgentError.format_error(error, include_trace=include_trace)
        prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

        if isinstance(error, (ValidationError, ValueError)):
            logger.error(f'{prefix}{error_msg}')
            if 'Max token limit reached' in error_msg:
                # cut tokens from history
                self.message_manager.max_input_tokens = self.max_input_tokens - 500
                logger.info(f'Cutting tokens from history - new max input tokens: {self.message_manager.max_input_tokens}')
                self.message_manager.cut_messages()
            elif 'Could not parse response' in error_msg:
                # give model a hint how output should look like
                error_msg += '\n\nReturn a valid JSON object with the required fields.'

            self.consecutive_failures += 1
        elif isinstance(error, RateLimitError):
            logger.warning(f'{prefix}{error_msg}')
            await asyncio.sleep(self.retry_delay)
            self.consecutive_failures += 1
        else:
            logger.error(f'{prefix}{error_msg}')
            self.consecutive_failures += 1

        return [ActionResult(error=error_msg, include_in_memory=True)]

    def _make_history_item(
        self,
        model_output: AgentOutput | None,
        state: BrowserState,
        result: list[ActionResult],
    ) -> None:
        """Create and store a history item"""
        if model_output:
            interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
        else:
            interacted_elements = [None]

        state_history = BrowserStateHistory(
            url=state.url,
            title=state.title,
            tabs=state.tabs,
            interacted_element=interacted_elements,
            screenshot=state.screenshot,
        )

        history_item = AgentHistory(model_output=model_output, result=result, state=state_history)

        self.history.history.append(history_item)

    @time_execution_async('--get_next_action')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""

        if self.model_name == 'deepseek-reasoner':
            converted_input_messages = self.message_manager.convert_messages_for_non_function_calling_models(input_messages)
            merged_input_messages = self.message_manager.merge_successive_human_messages(converted_input_messages)
            output = self.llm.invoke(merged_input_messages)
            # TODO: currently invoke does not return reasoning_content, we should override invoke
            try:
                parsed_json = self.message_manager.extract_json_from_model_output(output.content)
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f'Failed to parse model output: {str(e)}')
                raise ValueError('Could not parse response.')
        elif self.tool_calling_method is None:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']
        else:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True, method=self.tool_calling_method)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']

        if parsed is None:
            logger.error(f"Could not parse response {response['raw']}: {response['parsing_error']}")
            raise ValueError('Could not parse response.')

        # Cut the number of actions to max_actions_per_step
        parsed.action = parsed.action[: self.max_actions_per_step]
        self._log_response(parsed)
        self.n_steps += 1

        return parsed

    def _log_response(self, response: AgentOutput) -> None:
        """Log the model's response"""
        if 'Success' in response.current_state.evaluation_previous_goal:
            emoji = '👍'
        elif 'Failed' in response.current_state.evaluation_previous_goal:
            emoji = '⚠'
        else:
            emoji = '🤷'

        logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
        logger.info(f'🧠 Memory: {response.current_state.memory}')
        logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
        for i, action in enumerate(response.action):
            logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')

    def _save_conversation(self, input_messages: list[BaseMessage], response: Any) -> None:
        """Save conversation history to file if path is specified"""
        if not self.save_conversation_path:
            return

        # create folders if not exists
        os.makedirs(os.path.dirname(self.save_conversation_path), exist_ok=True)

        with open(
            self.save_conversation_path + f'_{self.n_steps}.txt',
            'w',
            encoding=self.save_conversation_path_encoding,
        ) as f:
            self._write_messages_to_file(f, input_messages)
            self._write_response_to_file(f, response)

    def _write_messages_to_file(self, f: Any, messages: list[BaseMessage]) -> None:
        """Write messages to conversation file"""
        for message in messages:
            f.write(f' {message.__class__.__name__} \n')

            if isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        f.write(item['text'].strip() + '\n')
            elif isinstance(message.content, str):
                try:
                    content = json.loads(message.content)
                    f.write(json.dumps(content, indent=2) + '\n')
                except json.JSONDecodeError:
                    f.write(message.content.strip() + '\n')

            f.write('\n')

    def _write_response_to_file(self, f: Any, response: Any) -> None:
        """Write model response to conversation file"""
        f.write(' RESPONSE\n')
        f.write(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

    def _log_agent_run(self) -> None:
        """Log the agent run for the current task"""
        logger.info(f'🚀 Starting task: {self.task}')

        logger.debug(f'Version: {self.version}, Source: {self.source}')
        self.telemetry.capture(
            AgentRunTelemetryEvent(
                agent_id=self.agent_id,
                use_vision=self.use_vision,
                task=self.task,
                model_name=self.model_name,
                chat_model_library=self.chat_model_library,
                version=self.version,
                source=self.source,
            )
        )

    @observe(name='agent.run')
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """
        Run the Agent. If multiple tasks were provided, they are completed sequentially.
        The Agent reuses the same browser context and memory across tasks by default.
        """

        overall_history = AgentHistoryList(history=[])
        try:
            while self.current_task_index < len(self.tasks):
                task = self.tasks[self.current_task_index]
                idx = self.current_task_index + 1
                logger.info('\n===============================')
                logger.info(f'Starting sub-task {idx}/{len(self.tasks)}:')
                logger.info(f'Task: {task}')
                logger.info('===============================\n')

                self.task = task  # update the internal "active" task
                self.current_task_index = 0  # Tracks which task is being executed

                if self.reset_messages_on_new_task:
                    self.message_manager.reset_messages(self.task)
                else:
                    self.message_manager.add_new_task(task)

                self._log_agent_run()

                # Reset step/failure counters and the per-task history
                # so each sub-task has fresh step counts but shared memory
                self.n_steps = 1
                self.consecutive_failures = 0
                self.history = AgentHistoryList(history=[])  # start fresh for each task

                # Execute initial actions for the first sub-task if provided
                if self.initial_actions and self.current_task_index == 0:
                    result = await self.controller.multi_act(
                        self.initial_actions,
                        self.browser_context,
                        check_for_new_elements=False,
                    )
                    self._last_result = result

                for _ in range(max_steps):
                    if self._too_many_failures():
                        break

                    # Check control flags before each step
                    if not await self._handle_control_flags():
                        break

                    await self.step()

                    if self.history.is_done():
                        # Optionally validate the output if requested
                        if self.validate_output and _ < max_steps - 1:
                            if not await self._validate_output():
                                continue

                        logger.info(f'✅ Sub-task completed successfully: {task}')
                        if self.register_done_callback:
                            self.register_done_callback(self.history)
                        break
                else:
                    logger.info(f'❌ Failed to complete sub-task in maximum steps: {task}')

                # Merge this sub-task's history into the overall history
                overall_history.history.extend(self.history.history)

                # Move on to next task
                self.current_task_index += 1

        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    success=overall_history.is_done(),
                    steps=self.n_steps,
                    max_steps_reached=self.n_steps >= max_steps,
                    errors=overall_history.errors(),
                )
            )

            # Cleanup (close the browser if it wasn't injected)
            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

            # Generate a GIF from the final (overall) history if requested
            if self.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.generate_gif, str):
                    output_path = self.generate_gif
                self.create_history_gif(output_path=output_path, show_task=True)

        return overall_history

    def _too_many_failures(self) -> bool:
        """Check if we should stop due to too many failures"""
        if self.consecutive_failures >= self.max_failures:
            logger.error(f'❌ Stopping due to {self.max_failures} consecutive failures')
            return True
        return False

    async def _handle_control_flags(self) -> bool:
        """Handle pause and stop flags. Returns True if execution should continue."""
        if self._stopped:
            logger.info('Agent stopped')
            return False

        while self._paused:
            await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
            if self._stopped:  # Allow stopping while paused
                return False
        return True

    async def _validate_output(self) -> bool:
        """Validate the output of the last action is what the user wanted"""
        validation_prompt = ValidatorSystemPrompt(self.task)

        if self.browser_context.session:
            state = await self.browser_context.get_state(use_vision=self.use_vision)
            content = AgentMessagePrompt(
                state=state,
                result=self._last_result,
                include_attributes=self.include_attributes,
                max_error_length=self.max_error_length,
            )
            msg = [validation_prompt.get_system_message(), content.get_user_message()]
        else:
            # if no browser session, we can't validate the output
            return True

        class ValidationResult(BaseModel):
            is_valid: bool
            reason: str

        validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
        response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
        parsed: ValidationResult = response['parsed']
        is_valid = parsed.is_valid
        if not is_valid:
            logger.info(f'❌ Validator decision: {parsed.reason}')
            msg = f'The output is not yet correct. {parsed.reason}.'
            self._last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
        else:
            logger.info(f'✅ Validator decision: {parsed.reason}')
        return is_valid

    async def rerun_history(
        self,
        history: AgentHistoryList,
        max_retries: int = 3,
        skip_failures: bool = True,
        delay_between_actions: float = 2.0,
    ) -> list[ActionResult]:
        """
        Rerun a saved history of actions with error handling and retry logic.

        Args:
            history: The history to replay
            max_retries: Maximum number of retries per action
            skip_failures: Whether to skip failed actions or stop execution
            delay_between_actions: Delay between actions in seconds

        Returns:
            List of action results
        """
        results = []

        for i, history_item in enumerate(history.history):
            goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
            logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

            if (
                not history_item.model_output
                or not history_item.model_output.action
                or history_item.model_output.action == [None]
            ):
                logger.warning(f'Step {i + 1}: No action to replay, skipping')
                results.append(ActionResult(error='No action to replay'))
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(history_item, delay_between_actions)
                    results.extend(result)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
                        logger.error(error_msg)
                        if not skip_failures:
                            results.append(ActionResult(error=error_msg))
                            raise RuntimeError(error_msg)
                    else:
                        logger.warning(f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...')
                        await asyncio.sleep(delay_between_actions)

        return results

    async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
        """Execute a single step from history with element validation"""
        state = await self.browser_context.get_state()
        if not state or not history_item.model_output:
            raise ValueError('Invalid state or model output')

        updated_actions = []
        for i, action in enumerate(history_item.model_output.action):
            updated_action = await self._update_action_indices(
                history_item.state.interacted_element[i],
                action,
                state,
            )
            updated_actions.append(updated_action)

            if updated_action is None:
                raise ValueError(f'Could not find matching element {i} in current page')

        result = await self.controller.multi_act(updated_actions, self.browser_context)
        await asyncio.sleep(delay)
        return result

    async def _update_action_indices(
        self,
        historical_element: Optional[DOMHistoryElement],
        action: ActionModel,
        current_state: BrowserState,
    ) -> Optional[ActionModel]:
        """
        Update action indices based on current page state.
        Returns updated action or None if element cannot be found.
        """
        if not historical_element or not current_state.element_tree:
            return action

        current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, current_state.element_tree)

        if not current_element or current_element.highlight_index is None:
            return None

        old_index = action.get_index()
        if old_index != current_element.highlight_index:
            action.set_index(current_element.highlight_index)
            logger.info(f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}')

        return action

    async def load_and_rerun(self, history_file: Optional[str | Path] = None, **kwargs) -> list[ActionResult]:
        """
        Load history from file and rerun it.

        Args:
            history_file: Path to the history file
            **kwargs: Additional arguments passed to rerun_history
        """
        if not history_file:
            history_file = 'AgentHistory.json'
        history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
        return await self.rerun_history(history, **kwargs)

    def save_history(self, file_path: Optional[str | Path] = None) -> None:
        """Save the current Agent history to a file"""
        if not file_path:
            file_path = 'AgentHistory.json'
        self.history.save_to_file(file_path)

    def create_history_gif(
        self,
        output_path: str = 'agent_history.gif',
        duration: int = 3000,
        show_goals: bool = True,
        show_task: bool = True,
        show_logo: bool = False,
        font_size: int = 40,
        title_font_size: int = 56,
        goal_font_size: int = 44,
        margin: int = 40,
        line_spacing: float = 1.5,
    ) -> None:
        """
        Create a GIF from the agent's entire (current) history
        with overlaid task/goal text.
        """
        if not self.history.history:
            logger.warning('No history to create GIF from')
            return

        images = []
        if not self.history.history[0].state.screenshot:
            logger.warning('No screenshot in first history item to create GIF')
            return

        # Attempt to load nicer fonts
        try:
            font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
            font_loaded = False
            for font_name in font_options:
                try:
                    if platform.system() == 'Windows':
                        # On Windows, you might need full paths to .ttf
                        font_name = os.path.join(
                            os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts'),
                            font_name + '.ttf',
                        )
                    regular_font = ImageFont.truetype(font_name, font_size)
                    title_font = ImageFont.truetype(font_name, title_font_size)
                    goal_font = ImageFont.truetype(font_name, goal_font_size)
                    font_loaded = True
                    break
                except OSError:
                    continue

            if not font_loaded:
                raise OSError('No preferred fonts found')
        except OSError:
            regular_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            goal_font = regular_font

        # Load logo if requested
        logo = None
        if show_logo:
            try:
                logo = Image.open('./static/openoperator.png')
                # Resize logo to ~150px height
                logo_height = 150
                aspect_ratio = logo.width / logo.height
                logo_width = int(logo_height * aspect_ratio)
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f'Could not load logo: {e}')

        # Optionally create an initial "task" frame
        if show_task and self.task:
            task_frame = self._create_task_frame(
                self.task,
                self.history.history[0].state.screenshot,
                title_font,
                regular_font,
                logo,
                line_spacing,
            )
            images.append(task_frame)

        # Process each history item
        for i, item in enumerate(self.history.history, 1):
            if not item.state.screenshot:
                continue

            img_data = base64.b64decode(item.state.screenshot)
            image = Image.open(io.BytesIO(img_data))
            if show_goals and item.model_output:
                image = self._add_overlay_to_image(
                    image=image,
                    step_number=i,
                    goal_text=item.model_output.current_state.next_goal,
                    regular_font=regular_font,
                    title_font=goal_font,
                    margin=margin,
                    logo=logo,
                )
            images.append(image)

        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )
            logger.info(f'Created GIF at {output_path}')
        else:
            logger.warning('No images found in history to create GIF')

    def _create_task_frame(
        self,
        task: str,
        first_screenshot: str,
        title_font: ImageFont.FreeTypeFont,
        regular_font: ImageFont.FreeTypeFont,
        logo: Optional[Image.Image] = None,
        line_spacing: float = 1.5,
    ) -> Image.Image:
        """Create initial frame for the GIF with the main task text"""
        img_data = base64.b64decode(first_screenshot)
        template = Image.open(io.BytesIO(img_data))
        image = Image.new('RGB', template.size, (0, 0, 0))
        draw = ImageDraw.Draw(image)

        center_y = image.height // 2
        margin = 140
        max_width = image.width - (2 * margin)

        # Slightly bigger font for the main task
        bigger_font = (
            ImageFont.truetype(regular_font.path, regular_font.size + 16) if hasattr(regular_font, 'path') else regular_font
        )

        wrapped_text = self._wrap_text(task, bigger_font, max_width)
        line_height = bigger_font.size * line_spacing
        lines = wrapped_text.split('\n')
        total_height = line_height * len(lines)

        text_y = center_y - (total_height / 2) + 50
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=bigger_font)
            text_x = (image.width - (bbox[2] - bbox[0])) // 2
            draw.text((text_x, text_y), line, font=bigger_font, fill=(255, 255, 255))
            text_y += line_height

        if logo:
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)

        return image

    def _add_overlay_to_image(
        self,
        image: Image.Image,
        step_number: int,
        goal_text: str,
        regular_font: ImageFont.FreeTypeFont,
        title_font: ImageFont.FreeTypeFont,
        margin: int,
        logo: Optional[Image.Image] = None,
        display_step: bool = True,
        text_color: tuple[int, int, int, int] = (255, 255, 255, 255),
        text_box_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> Image.Image:
        """Add step number and goal overlay to an image."""
        image = image.convert('RGBA')
        txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        if display_step:
            step_text = str(step_number)
            step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
            step_width = step_bbox[2] - step_bbox[0]
            step_height = step_bbox[3] - step_bbox[1]

            x_step = margin + 10
            y_step = image.height - margin - step_height - 10

            padding = 20
            step_bg_bbox = (
                x_step - padding,
                y_step - padding,
                x_step + step_width + padding,
                y_step + step_height + padding,
            )
            draw.rounded_rectangle(step_bg_bbox, radius=15, fill=text_box_color)

            draw.text((x_step, y_step), step_text, font=title_font, fill=text_color)

        max_width = image.width - (4 * margin)
        wrapped_goal = self._wrap_text(goal_text, title_font, max_width)
        goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=title_font)
        goal_width = goal_bbox[2] - goal_bbox[0]
        goal_height = goal_bbox[3] - goal_bbox[1]

        x_goal = (image.width - goal_width) // 2
        y_goal = y_step - goal_height - padding * 4

        padding_goal = 25
        goal_bg_bbox = (
            x_goal - padding_goal,
            y_goal - padding_goal,
            x_goal + goal_width + padding_goal,
            y_goal + goal_height + padding_goal,
        )
        draw.rounded_rectangle(goal_bg_bbox, radius=15, fill=text_box_color)

        draw.multiline_text(
            (x_goal, y_goal),
            wrapped_goal,
            font=title_font,
            fill=text_color,
            align='center',
        )

        if logo:
            logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
            txt_layer = Image.alpha_composite(logo_layer, txt_layer)

        result = Image.alpha_composite(image, txt_layer)
        return result.convert('RGB')

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        """Wrap text to fit within a given width."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            line = ' '.join(current_line)
            bbox = font.getbbox(line)
            if bbox[2] > max_width:
                if len(current_line) == 1:
                    lines.append(current_line.pop())
                else:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def add_task(self, task: str, index: Optional[int] = None) -> None:
        """
        Add a single new task to the tasks list.
        - Can only add tasks at or after the index of the current pending tasks.
        - If `index` is None, append to the end.
        """
        # The tasks up to `self.current_task_index` are considered completed (or in progress).
        # We only allow insertion at or after `self.current_task_index + 1`.
        if index is None:
            self.tasks.append(task)
            logger.info(f"Appended new task '{task}' at the end.")
        else:
            if index < self.current_task_index:
                raise ValueError('Cannot insert a new task before the completed or current task.')
            self.tasks.insert(index, task)
            logger.info(f"Inserted new task '{task}' at position {index}.")

    def add_tasks(self, tasks: List[str], index: Optional[int] = None) -> None:
        """
        Add multiple tasks to the tasks list.
        - Can only add tasks at or after the index of the current pending tasks.
        - If `index` is None, append them to the end.
        """
        if not tasks:
            logger.warning('No tasks provided to add.')
            return

        if index is None:
            self.tasks.extend(tasks)
            logger.info(f'Appended new tasks {tasks} at the end.')
        else:
            if index < self.current_task_index:
                raise ValueError('Cannot insert new tasks before the completed or current task.')
            for i, t in enumerate(tasks):
                self.tasks.insert(index + i, t)
            logger.info(f'Inserted new tasks {tasks} starting at position {index}.')

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
                raise ValueError('No pending tasks left to remove.')
            removed = self.tasks.pop()
            logger.info(f"Removed the last pending task '{removed}'.")
        else:
            if index < self.current_task_index or index >= len(self.tasks):
                raise ValueError('Index out of range or task is already completed or in progress.')
            removed = self.tasks.pop(index)
            logger.info(f"Removed task '{removed}' at index {index}.")

    def pause(self) -> None:
        """Pause the agent before the next step"""
        logger.info('🔄 pausing Agent')
        self._paused = True

    def resume(self) -> None:
        """Resume the agent"""
        logger.info('▶️ Agent resuming')
        self._paused = False

    def stop(self) -> None:
        """Stop the agent"""
        logger.info('⏹️ Agent stopping')
        self._stopped = True

    def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
        """Convert dictionary-based actions to ActionModel instances"""
        converted_actions = []
        action_model = self.ActionModel
        for action_dict in actions:
            # Each action_dict should have a single key-value pair
            action_name = next(iter(action_dict))
            params = action_dict[action_name]

            # Get the parameter model for this action from registry
            action_info = self.controller.registry.registry.actions[action_name]
            param_model = action_info.param_model

            # Create validated parameters
            validated_params = param_model(**params)

            # Create ActionModel instance
            action_model = self.ActionModel(**{action_name: validated_params})
            converted_actions.append(action_model)

        return converted_actions
