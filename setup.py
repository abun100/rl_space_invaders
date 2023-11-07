from setuptools import find_packages, setup

setup(
    name="final-project",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'setuptools',
    ],
    python_requires=">=3.10",
    version="1.0.0",
)