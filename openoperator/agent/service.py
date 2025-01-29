from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from lmnr import observe
from openai import RateLimitError
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
from openoperator.llm import LLM
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
        llm: LLM,
        browser: Browser | None = None,
        browser_context: BrowserContext | None = None,
        controller: Controller = Controller(),
        use_vision: bool = True,
        max_failures: int = 3,
        retry_delay: int = 10,
        system_prompt_class: Type[SystemPrompt] = SystemPrompt,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        initial_context: Optional[str] = None,
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

        self.tasks = []
        self.current_task_index = 0  # Tracks which task is being executed

        self.use_vision = use_vision
        self.llm = llm
        self._last_result = None
        self.include_attributes = include_attributes
        self.max_error_length = max_error_length

        # Controller setup
        self.controller = controller
        self.max_actions_per_step = max_actions_per_step

        # Browser setup
        self.injected_browser = browser is not None
        self.injected_browser_context = browser_context is not None
        self.initial_context = initial_context

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

        self.tool_calling_method = self._set_tool_calling_method(tool_calling_method)

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

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

    def _set_tool_calling_method(self, tool_calling_method: Optional[str]) -> Optional[str]:
        if tool_calling_method == 'auto':
            # Providers that support tool calling
            providers_with_function_calling = {
                'anthropic',  # ChatAnthropic
                'fireworks_ai',  # ChatFireworks
                'azure',  # AzureChatOpenAI
                'openai',  # ChatOpenAI
                'together_ai',  # ChatTogether
                'vertex_ai',  # ChatVertexAI
                'gemini',  # ChatGoogleGenerativeAI
                'groq',  # ChatGroq
                'bedrock',  # ChatBedrock
                'huggingface',  # ChatHuggingFace
                'ollama',  # ChatOllama
            }

            # Providers that support tool calling with JSON mode
            providers_with_json_mode = {
                'openai',  # ChatOpenAI
                'azure',  # AzureChatOpenAI
                'fireworks_ai',  # ChatFireworks
                'together_ai',  # ChatTogether
                'groq',  # ChatGroq
                'ollama',  # ChatOllama
            }

            # Determine the appropriate tool calling method
            if self.llm.model_provider in providers_with_json_mode:
                return 'json_mode'
            elif self.llm.model_provider in providers_with_function_calling:
                return 'function_calling'
            else:
                return None
        else:
            # Return the specified tool calling method if not set to 'auto'
            return tool_calling_method

    @time_execution_async('--step')
    async def _step(self, step_info: Optional[AgentStepInfo] = None) -> None:
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

        if self.llm.model_provider == 'deepseek' and self.llm.model_name == 'deepseek-reasoner':
            converted_input_messages = self.message_manager.convert_messages_for_non_function_calling_models(input_messages)
            merged_input_messages = self.message_manager.merge_successive_human_messages(converted_input_messages)
            output = self.llm.model.invoke(merged_input_messages)
            # TODO: currently invoke does not return reasoning_content, we should override invoke
            try:
                parsed_json = self.message_manager.extract_json_from_model_output(output.content)  # type: ignore
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f'Failed to parse model output: {str(e)}')
                raise ValueError('Could not parse response.')
        elif self.tool_calling_method is None:
            structured_llm = self.llm.model.with_structured_output(self.AgentOutput, include_raw=True)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']
        else:
            structured_llm = self.llm.model.with_structured_output(
                self.AgentOutput, include_raw=True, method=self.tool_calling_method
            )
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']

        if parsed is None:
            # Try manual parsing
            raw: BaseMessage = response['raw']
            logger.debug('Automatic response parsing failed. Trying manual parsing.')

            if not isinstance(raw.content, str):
                raise ValueError('Unexpected response content type.')

            try:
                obj = json.loads(raw.content)
                parameters = obj['parameters']
            except Exception as e:
                logger.error(f'{str(e)}')
                raise ValueError('Could not parse response.')

            try:
                parsed = AgentOutput.model_validate(parameters)
            except ValidationError as e:
                for error in e.errors():
                    logger.error(f"{error['loc']} {error['msg']}")
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

        if len(response.action) == 0:
            logger.info('🛠️  No Action selected')

        for i, action in enumerate(response.action):
            logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')

    def _log_agent_run(self, task: str) -> None:
        """Log the agent run for the current task"""
        logger.info(f'🚀 Starting task: {task}')

        logger.debug(f'Version: {self.version}, Source: {self.source}')
        self.telemetry.capture(
            AgentRunTelemetryEvent(
                agent_id=self.agent_id,
                use_vision=self.use_vision,
                task=task,
                model_provider=self.llm.model_provider,
                model_name=self.llm.model_name,
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

        if len(self.tasks) == 0:
            raise Exception('You must add at least one task to the agent.')

        self.message_manager = MessageManager(
            task=self.tasks[0],
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            initial_context=self.initial_context,
        )

        overall_history = AgentHistoryList(history=[])
        try:
            while self.current_task_index < len(self.tasks):
                task = self.tasks[self.current_task_index]
                idx = self.current_task_index + 1
                logger.info('\n===============================')
                logger.info(f'Starting task {idx}/{len(self.tasks)}:')
                logger.info(f'Task: {task}')
                logger.info('===============================\n')

                if self.reset_messages_on_new_task:
                    self.message_manager.reset_messages(task)
                else:
                    self.message_manager.add_new_task(task)

                self._log_agent_run(task)

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

                    await self._step()

                    if self.history.is_done():
                        # Optionally validate the output if requested
                        if self.validate_output and _ < max_steps - 1:
                            if not await self._validate_output(task):
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

    async def _validate_output(self, task: str) -> bool:
        """Validate the output of the last action is what the user wanted"""
        validation_prompt = ValidatorSystemPrompt(task)

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

        validator = self.llm.model.with_structured_output(ValidationResult, include_raw=True)
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
