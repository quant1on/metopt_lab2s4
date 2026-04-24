class OptimizationError(Exception):
    """Base exception for optimization-specific errors."""


class ValidationError(OptimizationError):
    """Raised when function inputs or configuration are invalid."""
