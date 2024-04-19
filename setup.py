from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description_text = f.read()

# sample setup.py: https://github.com/pypa/sampleproject/blob/db5806e0a3204034c51b1c00dde7d5eb3fa2532e/setup.py
setup(
    name='datadrivenquadrature',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'xarray',
        'cvxpy'
    ],
    url="https://github.com/LDEO-CREW/data-driven-quadrature",
    author="Neal Ma",
    long_description=description_text
)