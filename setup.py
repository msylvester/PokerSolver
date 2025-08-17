from setuptools import setup, find_packages

setup(
    name="student-of-games",
    version="0.1.0",
    description="Implementation of Student of Games: A unified learning algorithm for both perfect and imperfect information games",
    author="Student of Games Implementation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "dataclasses-json>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ]
    },
)