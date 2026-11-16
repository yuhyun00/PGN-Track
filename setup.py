"""Installs the `pgntrack` package (tools/ run from the repository root)."""
from setuptools import find_packages, setup

setup(
    name='pgntrack',
    version='1.0.0',
    description='PGN-Track: a photometric Gauss-Newton motion stage for sports multi-object tracking',
    packages=find_packages(include=['pgntrack', 'pgntrack.*']),
    python_requires='>=3.9',
    install_requires=['numpy', 'scipy', 'opencv-python', 'lap', 'PyYAML', 'torch'],
    extras_require={'eval': ['trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git']},
)
