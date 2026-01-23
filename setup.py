from setuptools import setup, find_packages

def read_reqs(path):
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

requirements = read_reqs("requirements.txt")

setup(
    name="trustifai",
    version="0.1.0",
    description="Trustifai: A Comprehensive Framework for Evaluating AI Trustworthiness",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="Aaryan Verma",
    url="https://github.com/Aaryanverma/trustifai",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License"
    ],
)