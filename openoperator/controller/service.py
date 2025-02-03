import asyncio
import json
import logging
import re
from typing import Dict, Optional, Type

from main_content_extractor import MainContentExtractor
from pydantic import BaseModel

from openoperator.agent.views import ActionModel, ActionResult
from openoperator.browser.context import BrowserContext
from openoperator.controller.registry.service import Registry
from openoperator.controller.views import (
    ClickElementAction,
    DoneAction,
    GetDropdownOptionsAction,
    GetPageContentAction,
    GoToUrlAction,
    InputTextAction,
    NoParamsAction,
    OpenTabAction,
    ScrollAction,
    ScrollToTextAction,
    SearchGoogleAction,
    SelectDropdownOptionAction,
    SendKeysAction,
    SwitchTabAction,
    UploadFilesAction,
)
from openoperator.utils import time_execution_async, time_execution_sync

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: Optional[Type[BaseModel]] = None,
    ):
        self.exclude_actions = exclude_actions
        self.output_model = output_model
        self.registry = Registry(exclude_actions)
        self._register_default_actions()

    def _register_default_actions(self):
        """Register all default browser actions"""

        # Basic Navigation Actions
        @self.registry.action(
            'Search Google in the current tab',
            param_model=SearchGoogleAction,
            requires_browser=True,
        )
        async def search_google(
            params: SearchGoogleAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()
            await page.goto(f'https://www.google.com/search?q={params.query}&udm=14')
            await page.wait_for_load_state()
            msg = f'🔍  Searched for "{params.query}" in Google'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            'Navigate to URL in the current tab',
            param_model=GoToUrlAction,
            requires_browser=True,
        )
        async def go_to_url(
            params: GoToUrlAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()
            await page.goto(params.url)
            await page.wait_for_load_state()
            msg = f'🔗  Navigated to {params.url}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            'Go back',
            param_model=NoParamsAction,
            requires_browser=True,
        )
        async def go_back(
            _: NoParamsAction,
            browser: BrowserContext,
        ):
            await browser.go_back()
            msg = '🔙  Navigated back'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Element Interaction Actions
        @self.registry.action(
            'Click element',
            param_model=ClickElementAction,
            requires_browser=True,
        )
        async def click_element(
            params: ClickElementAction,
            browser: BrowserContext,
        ):
            session = await browser.get_session()
            state = session.cached_state

            if params.index not in state.selector_map:
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = state.selector_map[params.index]
            initial_pages = len(session.context.pages)

            # if element has file uploader then dont click
            if await browser.is_file_uploader(element_node):
                msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            msg = None

            try:
                download_path = await browser.click_element_node(element_node)

                msg = f'🖱️  Clicked element with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

                if download_path:
                    msg = f'💾  Downloaded file to {download_path}'

                logger.info(msg)
                logger.debug(f'Element xpath: {element_node.xpath}')
                if len(session.context.pages) > initial_pages:
                    new_tab_msg = 'New tab opened - switching to it'
                    msg += f' - {new_tab_msg}'
                    logger.info(new_tab_msg)
                    await browser.switch_to_tab(-1)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
                return ActionResult(error=str(e))

        @self.registry.action(
            'Input text into an input interactive element. Allows placeholders.',
            param_model=InputTextAction,
            requires_browser=True,
            allows_additional_information=True,
        )
        async def input_text(
            params: InputTextAction,
            browser: BrowserContext,
            additional_information: Dict[str, str] = {},
        ):
            session = await browser.get_session()
            state = session.cached_state

            if params.index not in state.selector_map:
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            element_node = state.selector_map[params.index]

            # Function to replace placeholders with actual values and collect missing keys
            def replace_placeholders(text: str, info: Dict[str, str]) -> str:
                pattern = re.compile(r'\*\*\*(.*?)\*\*\*')
                placeholders = pattern.findall(text)
                missing_keys = [key for key in placeholders if key not in info]

                if missing_keys:
                    # Remove duplicates by converting to a set
                    unique_missing_keys = set(missing_keys)
                    missing_keys_str = ', '.join(f'"{key}"' for key in unique_missing_keys)
                    raise KeyError(f'Placeholder key(s) {missing_keys_str} not found in additional_information.')

                # If all keys are present, perform the replacement
                def replacer(match):
                    key = match.group(1)
                    return info[key]

                return pattern.sub(replacer, text)

            try:
                # Replace placeholders in the input text
                processed_text = replace_placeholders(params.text, additional_information)
            except KeyError as e:
                logger.error(str(e))
                raise Exception(f'Placeholder replacement error: {str(e)}')

            # Input the processed text into the element
            await browser.input_text_element_node(element_node, processed_text)

            msg = f'⌨️  Input "{processed_text}" into index {params.index}'
            logger.info(msg)
            logger.debug(f'Element xpath: {element_node.xpath}')
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Tab Management Actions
        @self.registry.action('Switch tab', param_model=SwitchTabAction, requires_browser=True)
        async def switch_tab(
            params: SwitchTabAction,
            browser: BrowserContext,
        ):
            await browser.switch_to_tab(params.page_id)
            # Wait for tab to be ready
            page = await browser.get_current_page()
            await page.wait_for_load_state()
            msg = f'🔄  Switched to tab {params.page_id}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action('Open url in new tab', param_model=OpenTabAction, requires_browser=True)
        async def open_tab(
            params: OpenTabAction,
            browser: BrowserContext,
        ):
            await browser.create_new_tab(params.url)
            msg = f'🔗  Opened new tab with {params.url}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Content Actions
        @self.registry.action(
            'Get page content as pure text or markdown with links if include_links is set to true',
            param_model=GetPageContentAction,
            requires_browser=True,
        )
        async def get_page_content(
            params: GetPageContentAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()
            output_format = 'markdown' if params.include_links else 'text'
            content = MainContentExtractor.extract(  # type: ignore
                html=await page.content(),
                output_format=output_format,
            )
            msg = f'📄  Page content in {output_format} format\n: {content}\n'
            logger.info(msg)
            return ActionResult(extracted_content=msg)

        @self.registry.action(
            'Complete task',
            param_model=DoneAction,
            requires_browser=False,
        )
        async def done(params: DoneAction):
            return ActionResult(is_done=True, extracted_content=params.text)

        @self.registry.action(
            'Scroll down the page by pixel amount - if no amount is specified, scroll down one page',
            param_model=ScrollAction,
            requires_browser=True,
        )
        async def scroll_down(
            params: ScrollAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, {params.amount});')
            else:
                await page.keyboard.press('PageDown')

            amount = f'{params.amount} pixels' if params.amount is not None else 'one page'
            msg = f'🔍  Scrolled down the page by {amount}'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # scroll up
        @self.registry.action(
            'Scroll up the page by pixel amount - if no amount is specified, scroll up one page',
            param_model=ScrollAction,
            requires_browser=True,
        )
        async def scroll_up(
            params: ScrollAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, -{params.amount});')
            else:
                await page.keyboard.press('PageUp')

            amount = f'{params.amount} pixels' if params.amount is not None else 'one page'
            msg = f'🔍  Scrolled up the page by {amount}'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # send keys
        @self.registry.action(
            'Send strings of special keys like Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. Be aware of different operating systems and their shortcuts',
            param_model=SendKeysAction,
            requires_browser=True,
        )
        async def send_keys(
            params: SendKeysAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()

            await page.keyboard.press(params.keys)
            msg = f'⌨️  Sent keys: {params.keys}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            description='If you dont find something which you want to interact with, scroll to it',
            param_model=ScrollToTextAction,
            requires_browser=True,
        )
        async def scroll_to_text(
            params: ScrollToTextAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()

            try:
                # Try different locator strategies
                locators = [
                    page.get_by_text(params.text, exact=False),
                    page.locator(f'text={params.text}'),
                    page.locator(f"//*[contains(text(), '{params.text}')]"),
                ]

                for locator in locators:
                    try:
                        # First check if element exists and is visible
                        if await locator.count() > 0 and await locator.first.is_visible():
                            await locator.first.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)  # Wait for scroll to complete
                            msg = f'🔍  Scrolled to text: {params.text}'
                            logger.info(msg)
                            return ActionResult(extracted_content=msg, include_in_memory=True)
                    except Exception as e:
                        logger.debug(f'Locator attempt failed: {str(e)}')
                        continue

                msg = f"Text '{params.text}' not found or not visible on page"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                msg = f"Failed to scroll to text '{params.text}': {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg, include_in_memory=True)

        @self.registry.action(
            description='Get all options from a native dropdown',
            param_model=GetDropdownOptionsAction,
            requires_browser=True,
        )
        async def get_dropdown_options(
            params: GetDropdownOptionsAction,
            browser: BrowserContext,
        ) -> ActionResult:
            """Get all options from a native dropdown"""
            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[params.index]

            try:
                # Frame-aware approach since we know it works
                all_options = []
                frame_index = 0

                for frame in page.frames:
                    try:
                        options = await frame.evaluate(
                            """
                            (xpath) => {
                                const select = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!select) return null;

                                return {
                                    options: Array.from(select.options).map(opt => ({
                                        text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
                                        value: opt.value,
                                        index: opt.index
                                    })),
                                    id: select.id,
                                    name: select.name
                                };
                            }
                        """,
                            dom_element.xpath,
                        )

                        if options:
                            logger.debug(f'Found dropdown in frame {frame_index}')
                            logger.debug(f'Dropdown ID: {options["id"]}, Name: {options["name"]}')

                            formatted_options = []
                            for opt in options['options']:
                                # encoding ensures AI uses the exact string in select_dropdown_option
                                encoded_text = json.dumps(opt['text'])
                                formatted_options.append(f'{opt["index"]}: text={encoded_text}')

                            all_options.extend(formatted_options)

                    except Exception as frame_e:
                        logger.debug(f'Frame {frame_index} evaluation failed: {str(frame_e)}')

                    frame_index += 1

                if all_options:
                    msg = '\n'.join(all_options)
                    msg += '\nUse the exact text string in select_dropdown_option'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    msg = 'No options found in any frame for dropdown'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                logger.error(f'Failed to get dropdown options: {str(e)}')
                msg = f'Error getting options: {str(e)}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            description='Select dropdown option for interactive element index by the text of the option you want to select',
            param_model=SelectDropdownOptionAction,
            requires_browser=True,
        )
        async def select_dropdown_option(
            params: SelectDropdownOptionAction,
            browser: BrowserContext,
        ) -> ActionResult:
            """Select dropdown option by the text of the option you want to select"""
            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[params.index]

            # Validate that we're working with a select element
            if dom_element.tag_name != 'select':
                logger.error(f'Element is not a select! Tag: {dom_element.tag_name}, Attributes: {dom_element.attributes}')
                msg = f'Cannot select option: Element with index {params.index} is a {dom_element.tag_name}, not a select'
                return ActionResult(extracted_content=msg, include_in_memory=True)

            logger.debug(f"Attempting to select '{params.text}' using xpath: {dom_element.xpath}")
            logger.debug(f'Element attributes: {dom_element.attributes}')
            logger.debug(f'Element tag: {dom_element.tag_name}')

            try:
                frame_index = 0
                for frame in page.frames:
                    try:
                        logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

                        # First verify we can find the dropdown in this frame
                        find_dropdown_js = """
                            (xpath) => {
                                try {
                                    const select = document.evaluate(xpath, document, null,
                                        XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                    if (!select) return null;
                                    if (select.tagName.toLowerCase() !== 'select') {
                                        return {
                                            error: `Found element but it's a ${select.tagName}, not a SELECT`,
                                            found: false
                                        };
                                    }
                                    return {
                                        id: select.id,
                                        name: select.name,
                                        found: true,
                                        tagName: select.tagName,
                                        optionCount: select.options.length,
                                        currentValue: select.value,
                                        availableOptions: Array.from(select.options).map(o => o.text.trim())
                                    };
                                } catch (e) {
                                    return {error: e.toString(), found: false};
                                }
                            }
                        """

                        dropdown_info = await frame.evaluate(find_dropdown_js, dom_element.xpath)

                        if dropdown_info:
                            if not dropdown_info.get('found'):
                                logger.error(f'Frame {frame_index} error: {dropdown_info.get("error")}')
                                continue

                            logger.debug(f'Found dropdown in frame {frame_index}: {dropdown_info}')

                            # "label" because we are selecting by text
                            # nth(0) to disable error thrown by strict mode
                            # timeout=1000 because we are already waiting for all network events, therefore ideally we don't need to wait a lot here (default 30s)
                            selected_option_values = (
                                await frame.locator('//' + dom_element.xpath)
                                .nth(0)
                                .select_option(label=params.text, timeout=1000)
                            )

                            msg = f'selected option {params.text} with value {selected_option_values}'
                            logger.info(msg + f' in frame {frame_index}')

                            return ActionResult(extracted_content=msg, include_in_memory=True)

                    except Exception as frame_e:
                        logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
                        logger.error(f'Frame type: {type(frame)}')
                        logger.error(f'Frame URL: {frame.url}')

                    frame_index += 1

                msg = f"Could not select option '{params.text}' in any frame"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                msg = f'Selection failed: {str(e)}'
                logger.error(msg)
                return ActionResult(error=msg, include_in_memory=True)

        @self.registry.action(
            'Upload files', param_model=UploadFilesAction, requires_browser=True, allows_additional_information=True
        )
        async def upload_files(
            params: UploadFilesAction,
            browser: BrowserContext,
            additional_information: Dict[str, str] = {},
        ):
            index = params.index

            dom_el = await browser.get_dom_element_by_index(index)

            if dom_el is None:
                return ActionResult(error=f'No element found at index {index}')

            file_upload_dom_el = dom_el.get_file_upload_element()

            if file_upload_dom_el is None:
                logger.info(f'No file upload element found at index {index}')
                return ActionResult(error=f'No file upload element found at index {index}')

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)

            if file_upload_el is None:
                logger.info(f'No file upload element found at index {index}')
                return ActionResult(error=f'No file upload element found at index {index}')

            try:
                paths = []
                for file_name in params.file_names:
                    if file_name not in additional_information:
                        logger.warning(f"No mapping for {file_name} available.")
                        continue
                    paths.append(additional_information[file_name])
                await file_upload_el.set_input_files(paths)
                msg = f'Successfully uploaded files to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg)
            except Exception as e:
                logger.debug(f'Error in set_input_files: {str(e)}')
                return ActionResult(error=f'Failed to upload files to index {index}. {str(e)}')

        @self.registry.action(
            'Close file dialog',
            param_model=NoParamsAction,
            requires_browser=True,
        )
        async def close_file_dialog(
            params: NoParamsAction,
            browser: BrowserContext,
        ):
            page = await browser.get_current_page()
            await page.keyboard.press('Escape')

        def action(self, description: str, **kwargs):
            """Decorator for registering custom actions

            @param description: Describe the LLM what the function does (better description == better function calling)
            """
            return self.registry.action(description, **kwargs)

    @time_execution_async('--multi-act')
    async def multi_act(
        self,
        actions: list[ActionModel],
        browser_context: BrowserContext,
        additional_information: Optional[Dict[str, str]],
        check_for_new_elements: bool = True,
    ) -> list[ActionResult]:
        """Execute multiple actions"""
        results = []

        session = await browser_context.get_session()
        cached_selector_map = session.cached_state.selector_map
        cached_path_hashes = set(e.hash.branch_path_hash for e in cached_selector_map.values())
        await browser_context.remove_highlights()

        for i, action in enumerate(actions):
            if action.get_index() is not None and i != 0:
                new_state = await browser_context.get_state()
                new_path_hashes = set(e.hash.branch_path_hash for e in new_state.selector_map.values())
                if check_for_new_elements and not new_path_hashes.issubset(cached_path_hashes):
                    # next action requires index but there are new elements on the page
                    logger.info(f'Something new appeared after action {i} / {len(actions)}')
                    break

            results.append(await self.act(action, browser_context, additional_information))

            logger.debug(f'Executed action {i + 1} / {len(actions)}')
            if results[-1].is_done or results[-1].error or i == len(actions) - 1:
                break

            await asyncio.sleep(browser_context.config.wait_between_actions)
            # hash all elements. if it is a subset of cached_state its fine - else break (new elements on page)

        return results

    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        browser_context: BrowserContext,
        additional_information: Optional[Dict[str, str]] = None,
    ) -> ActionResult:
        """Execute an action"""
        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    result = await self.registry.execute_action(
                        action_name, params, browser=browser_context, additional_information=additional_information
                    )
                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e
