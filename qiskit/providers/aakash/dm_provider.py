from qiskit.providers.basic_provider import BasicProvider
from .dm_simulator import DmSimulatorPy
from qiskit.providers.backend_compat import BackendV2Converter

class DmProvider(BasicProvider):

    @staticmethod
    def _deprecated_backend_names():
        names = BasicProvider._deprecated_backend_names()
        names["dm_simulator_py"] = "dm_simulator"
        return names

    def _verify_backends(self):
        verified = super()._verify_backends()
        backend_instance = BackendV2Converter(DmSimulatorPy())
        backend_name = backend_instance.name
        verified[backend_name] = backend_instance
        return verified
