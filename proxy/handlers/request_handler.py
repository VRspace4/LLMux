"""
Request preparation and conversion logic.
"""
import logging
from typing import Dict, Any, List

from openai_compat import convert_openai_request_to_anthropic
from anthropic import (
    sanitize_anthropic_request,
    inject_claude_code_system_message,
    add_prompt_caching,
)
from anthropic.thinking_keywords import process_thinking_keywords
from proxy.thinking_storage import inject_thinking_blocks

logger = logging.getLogger(__name__)


def strip_thinking_blocks_from_messages(messages: List[Dict[str, Any]], request_id: str = "") -> List[Dict[str, Any]]:
    """
    Strip thinking and redacted_thinking blocks from assistant messages.

    This is necessary when thinking is disabled but messages contain thinking blocks
    from previous responses (e.g., when switching models or disabling thinking).
    Anthropic API rejects: "When thinking is disabled, an assistant message in the
    final position cannot contain thinking"

    Args:
        messages: List of message dicts
        request_id: Request ID for logging

    Returns:
        Messages with thinking blocks removed from assistant messages
    """
    updated_messages = []
    stripped_count = 0

    for message in messages:
        if message.get("role") != "assistant":
            updated_messages.append(message)
            continue

        content = message.get("content")

        # If content is not a list, keep as-is (string content doesn't have thinking blocks)
        if not isinstance(content, list):
            updated_messages.append(message)
            continue

        # Filter out thinking blocks
        filtered_content = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("thinking", "redacted_thinking"):
                    stripped_count += 1
                    continue
            filtered_content.append(block)

        if filtered_content != content:
            # Content was modified, create new message
            new_message = message.copy()
            new_message["content"] = filtered_content
            updated_messages.append(new_message)
        else:
            updated_messages.append(message)

    if stripped_count > 0:
        logger.info(f"[{request_id}] Stripped {stripped_count} thinking blocks from messages (thinking disabled)")

    return updated_messages


def prepare_anthropic_request(
    openai_request: Dict[str, Any],
    request_id: str,
    is_native_anthropic: bool = False
) -> Dict[str, Any]:
    """
    Prepare an Anthropic API request from OpenAI or native format.

    Args:
        openai_request: The request data (OpenAI or Anthropic format)
        request_id: Request ID for logging
        is_native_anthropic: If True, skip OpenAI conversion

    Returns:
        Prepared Anthropic request dict
    """
    # Convert from OpenAI format if needed
    if not is_native_anthropic:
        anthropic_request = convert_openai_request_to_anthropic(openai_request)
    else:
        anthropic_request = openai_request.copy()

    # Process thinking keywords in messages (detect, strip, and get config)
    messages = anthropic_request.get("messages", [])
    processed_messages, thinking_config = process_thinking_keywords(messages)

    # Check if this is a tool use continuation (assistant message with tool_use)
    has_tool_use_in_assistant = False
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        has_tool_use_in_assistant = True
                        break

    if thinking_config:
        anthropic_request["messages"] = processed_messages
        # Only set thinking if not already configured AND not in tool use continuation
        if not anthropic_request.get("thinking") and not has_tool_use_in_assistant:
            anthropic_request["thinking"] = thinking_config
            logger.info(f"[{request_id}] Injected thinking config from keyword: {thinking_config}")
        elif has_tool_use_in_assistant:
            logger.info(f"[{request_id}] Skipping thinking injection due to tool use continuation")
        else:
            # Update budget if keyword specifies higher budget
            existing_budget = anthropic_request["thinking"].get("budget_tokens", 0)
            keyword_budget = thinking_config.get("budget_tokens", 0)
            if keyword_budget > existing_budget:
                anthropic_request["thinking"]["budget_tokens"] = keyword_budget
                logger.info(f"[{request_id}] Updated thinking budget from {existing_budget} to {keyword_budget}")

    # Ensure max_tokens is sufficient if thinking is enabled
    thinking = anthropic_request.get("thinking")
    if thinking and thinking.get("type") == "enabled":
        thinking_budget = thinking.get("budget_tokens", 16000)
        min_response_tokens = 1024
        required_total = thinking_budget + min_response_tokens
        if anthropic_request["max_tokens"] < required_total:
            anthropic_request["max_tokens"] = required_total
            logger.debug(
                f"[{request_id}] Increased max_tokens to {required_total} "
                f"(thinking: {thinking_budget} + response: {min_response_tokens})"
            )

        # Inject stored thinking blocks from previous responses
        anthropic_request["messages"] = inject_thinking_blocks(anthropic_request["messages"])
        logger.debug(f"[{request_id}] Injected stored thinking blocks if available")
    else:
        # When thinking is disabled, strip any thinking blocks from messages
        # This prevents "When thinking is disabled, an assistant message cannot contain thinking" errors
        anthropic_request["messages"] = strip_thinking_blocks_from_messages(
            anthropic_request["messages"], request_id
        )

    # Sanitize request for Anthropic API constraints
    anthropic_request = sanitize_anthropic_request(anthropic_request)

    # Inject Claude Code system message to bypass authentication detection
    anthropic_request = inject_claude_code_system_message(anthropic_request)

    # Add cache_control to message content blocks for optimal caching
    anthropic_request = add_prompt_caching(anthropic_request)

    return anthropic_request
