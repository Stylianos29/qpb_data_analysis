from setuptools import setup, find_packages # type: ignore

setup(
    name="qpb_data_analysis",
    version="0.1",
    description="A Python project for data analysis and processing of qpb data.",
    author="Stylianos Gregoriou",
    author_email="s.gregoriou@cyi.ac.cy",
    url="https://github.com/Stylianos29/qpb_data_analysis",
    # Tell setuptools to look in the 'core' directory for packages
    packages=find_packages(where="core"), 
    package_dir={"": "core"},  # The root of the package is now 'core'
    install_requires=[  # List your dependencies here
        "click",
        "gvar",
        "lsqfit",
        "h5py",
    ]
)
