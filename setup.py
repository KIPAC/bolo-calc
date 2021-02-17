from setuptools import setup

from python.bolo import version

setup(
    name="bolo",
    version=version.get_git_version(),
    author="",
    author_email="",
    url = "https://github.com/KIPAC/bolo-calc",
    package_dir={"":"python"},
    packages=["bolo"],
    description="Bolometric Calculations for CMB S4",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=['numpy', 'jax', 'jaxlib', 'pyyaml', 'cfgmdl'],
)
