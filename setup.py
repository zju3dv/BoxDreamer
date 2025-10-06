from setuptools import setup, find_packages
import os
from pathlib import Path


# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip()
                and not line.startswith("#")
                and not line.startswith("git+")
            ]
    return requirements


setup(
    name="boxdreamer",
    version="0.1.0",
    author="Yuanhong Yu",
    author_email="yuanhongyu.me@gmail.com",
    description="BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zju3dv/BoxDreamer",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
        "src": ["**/*.yaml", "**/*.json"],
        "src.demo": ["configs/*.yaml"],
        "src.demo.configs": ["*.yaml"],
    },
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "boxdreamer-cli=src.demo.demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "demo": [
            "gradio>=4.0",
            "decord",
            "PyQt5",
        ],
    },
)
