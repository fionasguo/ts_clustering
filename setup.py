from setuptools import find_packages, setup

setup(
    name="user-ts-clustering",
    version=
    "0.1",
    author=
    "Siyi Guo, Department of Computer Science, University of Southern California",
    author_email="<fionasguo@gmail.com>",
    description=
    "User Timeline Embedding",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    python_requires=">=3.9.13",
    install_requires=open("requirements.txt","r").read().splitlines(),
)