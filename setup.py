"""
distutils/setuptools install script.
"""
import re

from setuptools import find_packages, setup

VERSION_RE = re.compile(r"""([0-9dev.]+)""")
EDITABLE_INSTALL_RE = re.compile(r"""^-e\s+(.+)""")


def get_version():
    with open("VERSION", "r") as fh:
        init = fh.read().strip()
    return VERSION_RE.search(init).group(1)


def requirement_specification(requirements_line):
    editable_requirement = EDITABLE_INSTALL_RE.match(requirements_line)
    if editable_requirement is None:
        return requirements_line.strip()
    else:
        path = editable_requirement.group(1)
        return path.split("/")[-1]


def get_requirements():
    with open("requirements.txt", "r") as f:
        lines = f.read().splitlines()
        return [requirement_specification(l) for l in lines]


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="fedsom",
    version=get_version(),
    description="fedsom implementation",
    long_description=readme(),
    url="https://github.com/fed-som/fedsom",
    maintainer_email="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    # reference our temp script from above
    package_data={
        "fedsom": [
            "VERSION",
            "requirements.txt",
            "MANIFEST.in",
        ]
    },
    install_requires=get_requirements(),
)
