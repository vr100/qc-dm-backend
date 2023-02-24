from .dm_simulator import DmSimulatorPy
from .exceptions import DmError
from .dm_provider import DmProvider

# global instance for getting the desired backend
AAKASH_DM = DmProvider()
