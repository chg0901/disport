from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="disprot-prediction",
    version="0.1.0",
    author="DisProt Prediction Model Contributors",
    author_email="your.email@example.com",
    description="无序蛋白质区域预测模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/disprot-prediction",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "disprot-train=scripts.train:main",
            "disprot-predict=scripts.predict:main",
        ],
    },
) 