"""
Exceptions raised by DM backend
"""

from qiskit.exceptions import QiskitError


class DmError(QiskitError):
    """Base class for errors raised by DM  backend"""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
