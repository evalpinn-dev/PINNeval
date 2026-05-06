from setuptools import setup, find_packages
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    README = open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8").read()
except IOError:
    README = ""

setup(
    name="pinneval",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "absl-py",
        "flax",
        "jax",
        "jaxlib",
        "matplotlib",
        "ml-collections",
        "numpy",
        "optax",
        "scipy",
        "soap-jax @ git+https://github.com/haydn-jones/SOAP_JAX@v0.1.0",
        "tabulate",
        "wandb",
    ],
    extras_require={
        "testing": ["pytest"],
    },
    license="Apache-2.0",
    description="A library for evaluation PINNs following a comprehensive definition and implemented in JAX Flax.",
    long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
)
