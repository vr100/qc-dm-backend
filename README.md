# qc-dm-backend

The objective is to compile [qiskit-aakash git repo](https://github.com/indian-institute-of-science-qc/qiskit-aakash) as a separate python package similar to [IBMQ backend](https://github.com/Qiskit/qiskit-ibmq-provider)

# Usage instructions

## create conda environment

```bash
conda create -n aakash-dm python=3
conda activate aakash-dm
```

## build and install

```bash
python3 setup.py bdist_wheel
python3 -m pip install dist/qc_dm_backend-0.1-py3-none-any.whl
```

## test circuits

```bash
cd aakash/circuits/
python3 grover.py
```

# License note

The original repo  [qiskit-aakash git repo](https://github.com/indian-institute-of-science-qc/qiskit-aakash) is licensed under Apache License 2.0

And my compilation related changes are licensed under MIT License
