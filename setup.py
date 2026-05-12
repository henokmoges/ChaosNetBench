from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chaosnetbench",
    version="1.0.0",
    description="Benchmarking Spatio-Temporal Graph Neural Networks on Chaotic Lattice Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="H. T. Moges",
    author_email="ht.moges@gmail.com",
    license="MIT",
    url="https://github.com/htmoges/ChaosNetBench",
    project_urls={
        "Homepage": "https://htmoges.github.io",
        "Dataset": "https://huggingface.co/datasets/htmoges/chaosnetbench-cml",
    },
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0.0",
        "h5py>=3.8.0",
        "torch>=2.2.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
