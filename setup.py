''' Qiskit aakash density matrix backend '''

import os

from setuptools import setup

requirements = [
    "qiskit~=1.4.0",
    "matplotlib>=3.7",
    "scipy~=1.0",
    "sympy~=1.0"
]

readme_path = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'README.md')
with open(readme_path) as readme_file:
    long_desc = readme_file.read()


setup(
    name="qc-dm-backend",
    version="0.1",
    description="Qiskit aakash backend using density matrix",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url="https://github.com/vr100/qc-dm-backend",
    author="vr100",
    author_email="vr100@@",
    license="Apache 2.0",
    keywords="qiskit aakash backend",
    packages=['qiskit.providers.aakash',
        'qiskit.circuit.aakash'],
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.9"
)