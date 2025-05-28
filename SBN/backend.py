from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeFez, FakeMarrakesh
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService

def get_sampler(backend='aer', method='automatic'):
    """
    Returns a sampler of the specific type
    :param backend: The type of backend, one of:
        - 'aer': Noiseless Aer simulator
        - 'sherbrooke': Noisy simulator based on the Sherbrooke backend
        - 'fez': Noisy simulator based on the Fez backend
        - 'ibm': IBM Sherbrooke backend
    :param method: The method to be used for Aer
    :return: (Sampler, Backend, config | None)
    """

    if backend == 'aer':
        print("Using Noiseless Aer")
        simulator = AerSimulator(method=method)
        return Sampler(simulator), simulator, None
    elif backend == 'ibm_sherbrooke':
        print("Using IBM Sherbrooke backend")
        service = QiskitRuntimeService()
        backend = service.backend('ibm_sherbrooke')
        sampler = Sampler(backend)
        config = backend.configuration()
        return sampler, backend, config
    elif backend == 'ibm_torino':
        print("Using IBM Torino backend")
        service = QiskitRuntimeService()
        backend = service.backend('ibm_torino')
        sampler = Sampler(backend)
        config = backend.configuration()
        return sampler, backend, config
    elif backend == 'sherbrooke':
        print("Using Noisy Sherbrooke simulator")
        fake = FakeSherbrooke()
        simulator = AerSimulator(method=method).from_backend(fake)
        sampler = Sampler(simulator)
        config = fake.configuration()
        return sampler, simulator, config
    elif backend == 'fez':
        print("Using Noisy Fez simulator")
        fake = FakeFez()
        simulator = AerSimulator(method=method).from_backend(fake)
        sampler = Sampler(simulator)
        config = fake.configuration()
        return sampler, simulator, config
    elif backend == 'marrakesh':
        print("Using Noisy Marrakesh simulator")
        fake = FakeMarrakesh()
        simulator = AerSimulator(method=method).from_backend(fake)
        sampler = Sampler(simulator)
        config = fake.configuration()
        return sampler, simulator, config
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from 'aer', 'sherbrooke', 'fez', or 'ibm'.")