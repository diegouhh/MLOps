class NeuroOpsError(Exception):
    """Base error safe to expose through the API."""


class ValidationError(NeuroOpsError):
    """Invalid dataset, configuration, or prediction payload."""


class ResourceNotFoundError(NeuroOpsError):
    """Requested domain object does not exist."""


class PluginUnavailableError(NeuroOpsError):
    """An optional plugin cannot run in the active environment."""
