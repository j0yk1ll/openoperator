from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
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
from openoperator.agent.task_manager.service import TaskManager
from openoperator.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
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
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        tool_calling_method: Optional[str] = 'auto',
        reset_messages_on_new_task: bool = True,
        # Callbacks
        on_step: Callable[['BrowserState', 'AgentOutput', int], None] | None = None,
        on_done: Callable[['AgentHistoryList'], None] | None = None,
    ):
        # Unique ID for the agent
        self.agent_id = str(uuid.uuid4())

        # Set up the task manager
        self.task_manager = TaskManager()

        # Set up message manager
        self.message_manager = MessageManager(
            action_descriptions=controller.registry.get_prompt_description(),
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            initial_context=initial_context,
        )

        # Prevent downstream errors if the model does not support vision
        models_without_vision = ['deepseek-chat', 'deepseek-reasoner', 'llama-3.3-70b-versatile', 'deepseek-r1-distill-llama-70b']
        if llm.model_name in models_without_vision:
            logger.info('Disabled vision because the model does not support it.')
            use_vision = False

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

        # Telemetry
        self.telemetry = ProductTelemetry()

        # Action & output models
        self._setup_action_models()
        self._set_version_and_source()
        self.max_input_tokens = max_input_tokens

        self.tool_calling_method = self._set_tool_calling_method(tool_calling_method)

        # Step callbacks
        self.on_step = on_step
        self.on_done = on_done

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

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

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

    def _set_tool_calling_method(self, tool_calling_method: Optional[str]) -> Optional[str]:
        if tool_calling_method == 'auto':
            providers_with_function_calling = {
                'anthropic',
                'fireworks_ai',
                'azure',
                'openai',
                'together_ai',
                'vertex_ai',
                'groq',
                'bedrock',
                'huggingface',
                'ollama',
            }
            if self.llm.model_provider in providers_with_function_calling:
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
        """Convert dictionary-based actions to ActionModel instances."""
        converted_actions = []
        action_model = self.ActionModel
        for action_dict in actions:
            # Each action_dict should have a single key-value pair
            action_name = next(iter(action_dict))
            params = action_dict[action_name]

            # Grab the parameter model from the registry
            action_info = self.controller.registry.registry.actions[action_name]
            param_model = action_info.param_model

            # Create validated parameters
            validated_params = param_model(**params)

            # Create ActionModel instance
            action_model = self.ActionModel(**{action_name: validated_params})
            converted_actions.append(action_model)
        return converted_actions

    def _remove_think_tags(self, text: str) -> str:
        """Remove <think> tags from text (if using a model that includes them)."""
        think_tags = re.compile(r'<think>.*?</think>', re.DOTALL)
        return re.sub(think_tags, '', text)

    @time_execution_async('--get_next_action')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """
        Get the next action from the LLM based on the input messages.
        This method delegates to the LLM and attempts to parse a structured output.
        """
        if self.llm.model_provider == 'deepseek':
            converted_input_messages = self.message_manager.convert_messages_for_non_function_calling_models(input_messages)
            merged_input_messages = self.message_manager.merge_successive_human_messages(converted_input_messages)
            output = self.llm.model.invoke(merged_input_messages)
            if isinstance(output.content, str):
                output.content = self._remove_think_tags(output.content)
            try:
                parsed_json = self.message_manager.extract_json_from_model_output(output.content)  # type: ignore
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f'Failed to parse model output {output}: {str(e)}')
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
            # try manual parse
            raw = response['raw']
            if not isinstance(raw.content, str):
                raise ValueError('Unexpected response content type.')
            try:
                obj = json.loads(raw.content)
                parameters = obj['parameters']
            except Exception as e:
                logger.error(f'{str(e)}')
                raise ValueError('Could not parse response.')
            parsed = self.AgentOutput.model_validate(parameters)

        # Make sure we haven't gone over the max actions per step
        parsed.action = parsed.action[: self.max_actions_per_step]

        self._log_response(parsed)
        self.n_steps += 1
        return parsed

    def _log_response(self, response: AgentOutput) -> None:
        """Log the model's response at a high level."""
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
        else:
            for i, action in enumerate(response.action):
                logger.info(f'🛠️ Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')

    @time_execution_async('--step')
    async def _step(self, additional_information: Optional[Dict[str, str]] = None) -> None:
        """Execute a single step of the current task."""
        logger.info(f'\n📍 Step {self.n_steps}')
        state = None
        model_output = None
        result: list[ActionResult] = []

        try:
            # Grab state from browser
            state = await self.browser_context.get_state(use_vision=self.use_vision)

            if self._stopped or self._paused:
                logger.debug('Agent paused/stopped after getting state.')
                raise InterruptedError

            # Add the state to the message manager
            self.message_manager.add_state_message(state, self._last_result)
            input_messages = self.message_manager.get_messages()

            try:
                model_output = await self.get_next_action(input_messages)

                if self.on_step:
                    self.on_step(state, model_output, self.n_steps)

                # We remove the last state chunk from memory to keep token usage down
                self.message_manager._remove_last_state_message()

                if self._stopped or self._paused:
                    logger.debug('Agent paused/stopped after get_next_action.')
                    raise InterruptedError

                self.message_manager.add_model_output(model_output)
            except Exception as e:
                self.message_manager._remove_last_state_message()
                raise e

            # Execute the resulting actions
            result = await self.controller.multi_act(model_output.action, self.browser_context, additional_information)
            self._last_result = result

            # Reset consecutive failures if we succeed
            self.consecutive_failures = 0

        except InterruptedError:
            logger.debug('Agent paused or stopped.')
            return
        except Exception as e:
            # Handle an error during step
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
            if result and model_output and state:
                self._make_history_item(model_output, state, result)

    async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
        """Handle all types of errors that can occur during a step."""
        include_trace = logger.isEnabledFor(logging.DEBUG)
        error_msg = AgentError.format_error(error, include_trace=include_trace)
        prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

        if isinstance(error, (ValidationError, ValueError)):
            logger.error(f'{prefix}{error_msg}')
            if 'Max token limit reached' in error_msg:
                self.message_manager.max_input_tokens = self.max_input_tokens - 500
                logger.info(
                    f'Reducing max_input_tokens to {self.message_manager.max_input_tokens}. ' f'Attempting to trim history...'
                )
                self.message_manager.cut_messages()
            elif 'Could not parse response' in error_msg:
                # Give the model a hint about properly formatted JSON
                error_msg += '\n\nReturn a valid JSON object with the required fields.'
            self.consecutive_failures += 1

        elif isinstance(error, (RateLimitError, ResourceExhausted)):
            logger.warning(f'{prefix}{error_msg}')
            await asyncio.sleep(self.retry_delay)
            self.consecutive_failures += 1

        else:
            logger.error(f'{prefix}{error_msg}')
            self.consecutive_failures += 1

        return [ActionResult(error=error_msg, include_in_memory=True)]

    def _make_history_item(
        self,
        model_output: AgentOutput,
        state: BrowserState,
        result: list[ActionResult],
    ) -> None:
        """Create and store a history item for the current step."""
        interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
        state_history = BrowserStateHistory(
            url=state.url,
            title=state.title,
            tabs=state.tabs,
            interacted_element=interacted_elements,
            screenshot=state.screenshot,
        )
        history_item = AgentHistory(model_output=model_output, result=result, state=state_history)
        self.history.history.append(history_item)

    def _log_agent_run(self, task: str) -> None:
        """Log high-level info at the start of a task."""
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

    async def _validate_output(self, task: str) -> bool:
        """A post-step or post-task validation check using a validator prompt."""
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
        return True

    @observe(name='agent.run')
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """
        Run the Agent on all tasks in the TaskManager.
        By default, each task can take up to `max_steps` steps before aborting.
        """
        # Ensure we have tasks to run
        if not self.task_manager.has_tasks():
            raise Exception('You must add at least one task to the agent (TaskManager).')

        should_stop = False
        overall_history = AgentHistoryList(history=[])

        try:
            # Main loop over tasks
            while not self.task_manager.is_all_done() and not should_stop:
                current_task = self.task_manager.get_current_task()

                # suppress linter errors, should logically not happend
                if not current_task:
                    break

                idx = self.task_manager.current_task_index + 1
                logger.info('\n===============================')
                logger.info(f'Starting task {idx}/{len(self.task_manager.tasks)}:')
                logger.info(f'Task: {current_task.description}')
                logger.info('===============================\n')

                # Optionally reset or add new messages
                if self.reset_messages_on_new_task:
                    self.message_manager.reset_messages()

                # Add the task to the message manager
                self.message_manager.set_task(current_task)

                self._log_agent_run(current_task.description)

                # Reset step/failure counters for each new sub-task
                self.n_steps = 1
                self.consecutive_failures = 0
                self.history = AgentHistoryList(history=[])

                # Execute initial actions before the first task
                if self.initial_actions and self.task_manager.current_task_index == 0:
                    result = await self.controller.multi_act(
                        self.initial_actions, self.browser_context, additional_information=None, check_for_new_elements=False
                    )
                    self._last_result = result

                # Step loop for this particular task
                for _ in range(max_steps):
                    if self.consecutive_failures >= self.max_failures or self._stopped:
                        should_stop = True
                        break

                    # If paused, wait until resumed
                    while self._paused:
                        await asyncio.sleep(0.2)

                    await self._step(additional_information=current_task.additional_information)

                    if self.history.is_done():
                        # Optional: Validate the output
                        if self.validate_output and _ < max_steps - 1:
                            if not await self._validate_output(current_task.description):
                                # If validation failed, keep going
                                continue

                        logger.info(f'✅ Sub-task completed successfully: {current_task}')
                        if self.on_done:
                            self.on_done(self.history)
                        break
                else:
                    logger.info(f'❌ Failed to complete sub-task in max steps: {current_task}')

                # Merge this sub-task history
                overall_history.history.extend(self.history.history)
                # Move on to next task
                self.task_manager.increment_task_index()

        finally:
            # End-of-run telemetry
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    success=overall_history.is_done(),
                    steps=self.n_steps,
                    max_steps_reached=self.n_steps >= max_steps,
                    errors=overall_history.errors(),
                )
            )
            # Cleanup
            if not self.injected_browser_context:
                await self.browser_context.close()
            if not self.injected_browser and self.browser:
                await self.browser.close()

        return overall_history

    # ---------------------------------------------------------------------
    # Methods to delegate to TaskManager for tasks
    # ---------------------------------------------------------------------

    def add_task(self, task: str, additional_information: Optional[Dict[str, str]] = None, index: Optional[int] = None) -> None:
        """
        Add a single new task via the TaskManager.
        Optionally attach file references (unused here, but
        you could store them in a dictionary).
        """
        self.task_manager.add_task(task, additional_information=additional_information, index=index)

    def remove_task(self, index: Optional[int] = None) -> None:
        """
        Remove a single pending task from the TaskManager.
        """
        self.task_manager.remove_task(index=index)

    # ---------------------------------------------------------------------
    # Methods to control the agent state
    # ---------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the agent before the next step."""
        logger.info('🔄 Pausing Agent')
        self._paused = True

    def resume(self) -> None:
        """Resume the agent."""
        logger.info('▶️ Agent Resuming')
        self._paused = False

    def stop(self) -> None:
        """Stop the agent."""
        logger.info('⏹️ Agent Stopping')
        self._stopped = True
