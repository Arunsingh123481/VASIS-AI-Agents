# utils/exceptions.py
# Custom Exceptions for the CRDB Agentic Pipeline

class RequerySignal(Exception):
    """Raised when validation agents demand query refinement and re-retrieval."""
    def __init__(self, reason: str, refined_query: str):
        self.reason = reason
        self.refined_query = refined_query
        super().__init__(reason)


class ContradictionError(Exception):
    """Raised when severe factual contradictions are detected."""
    def __init__(self, details: list):
        self.details = details
        super().__init__(f"{len(details)} contradictions detected")


class IngestionError(Exception):
    """Raised when loading or parsing source documents fails."""
    pass


class IndexNotFoundError(Exception):
    """Raised when attempting to load a non-existent document index."""
    pass


class ModelError(Exception):
    """Raised when communicating with Ollama fails."""
    pass
