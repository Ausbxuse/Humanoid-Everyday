from setuptools import find_packages, setup

setup(
    name="humanoid_everyday",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "open3d",
        "tqdm",
    ],
    author="Zhenyu Zhao",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url="https://github.com/YourUser/HumanoidEveryday",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
