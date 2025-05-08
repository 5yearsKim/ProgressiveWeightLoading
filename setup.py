from setuptools import find_packages, setup

# Read requirements.txt into a list
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="pwl_model",
    version="0.1.0",
    description="Progressive Weight Loading",
    author="onion.kim",
    packages=find_packages(),  # Automatically find subpackages
    install_requires=[
        *requirements,
    ],
    python_requires=">=3.10",
)
