"""Exception types for pytans."""


class TansError(ValueError):
    """Base class for all pytans errors."""


class CorruptedDataError(TansError):
    """Raised when a compressed frame or bitstream fails validation."""
