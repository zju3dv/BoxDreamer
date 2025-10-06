from setuptools import setup, find_packages
import os


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="boxdreamer",
    version="0.1.0",
    author="Yuanhong Yu",
    author_email="yuanhongyu.me@gmail.com",
    description="BoxDreamer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zju3dv/BoxDreamer",
    packages=find_packages(),
    package_data={
        "src.demo": ["configs/*.yaml"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "boxdreamer-cli=src.demo.cli:main",
        ],
    },
)
