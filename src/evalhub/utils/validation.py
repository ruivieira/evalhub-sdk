"""Validation utilities for the EvalHub SDK."""

from typing import Any

from pydantic import BaseModel, ValidationError


def validate_config(config: dict[str, Any], config_class: type[BaseModel]) -> BaseModel:
    """Validate configuration data against a Pydantic model.

    Args:
        config: Configuration dictionary to validate
        config_class: Pydantic model class to validate against

    Returns:
        Validated configuration instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        return config_class(**config)
    except ValidationError as e:
        raise ValidationError(f"Configuration validation failed: {e}") from e


def validate_request(
    request: dict[str, Any], request_class: type[BaseModel]
) -> BaseModel:
    """Validate request data against a Pydantic model.

    Args:
        request: Request dictionary to validate
        request_class: Pydantic model class to validate against

    Returns:
        Validated request instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        return request_class(**request)
    except ValidationError as e:
        raise ValidationError(f"Request validation failed: {e}") from e
