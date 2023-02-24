from qiskit.providers.basicaer import BasicAerProvider
from .dm_simulator import DmSimulatorPy

class DmProvider(BasicAerProvider):

    @staticmethod
    def _deprecated_backend_names():
        names = BasicAerProvider._deprecated_backend_names()
        names["dm_simulator_py"] = "dm_simulator"
        return names

    def _verify_backends(self):
        verified = super()._verify_backends()
        backend_instance = self._get_backend_instance(DmSimulatorPy)
        backend_name = backend_instance.name()
        verified[backend_name] = backend_instance
        return verified
