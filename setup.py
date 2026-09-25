import os

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nureasoning-devkit",
    version="0.1.0",
    author="UCLA & Motional AD Inc.",
    author_email="nuScenes@motional.com",
    description="The devkit of the nuReasoning dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://huggingface.co/datasets/nureasoning/nuReasoning",
    python_requires=">=3.10",
    packages=setuptools.find_packages(include=["nureasoning", "nureasoning.*"]),
    include_package_data=True,
    package_data={"nureasoning.jepa_planning": ["configs/*.yaml"]},
    install_requires=[
        line.strip()
        for line in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        if line.strip() and not line.startswith(("#", "--"))
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "License :: Other/Proprietary License",
    ],
)
