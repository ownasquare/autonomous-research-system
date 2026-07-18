"""Typed, user-safe application errors."""


class ResearchSystemError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ResearchSystemError):
    """The requested mode cannot run with the available configuration."""


class SourceValidationError(ResearchSystemError):
    """A source or upload violates a validated ingestion boundary."""


class ProviderError(ResearchSystemError):
    """A source or model provider returned a typed failure."""


class CitationIntegrityError(ResearchSystemError):
    """A generated artifact refers to evidence that does not exist."""


class RunNotFoundError(ResearchSystemError):
    """A requested persisted run does not exist."""
