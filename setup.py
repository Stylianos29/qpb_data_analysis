from setuptools import setup, find_packages # type: ignore

setup(
    name="qpb_data_analysis",      # Package name
    version="0.1.0",               # Version
    description="Tools for processing and analyzing data files.",  # Description
    author="Stylianos Gregoriou",  # Author
    package_dir={"": "src"},       # Map the 'src/' directory as the package root
    packages=find_packages(where="src"),  # Find packages in 'src/'
    install_requires=[
        "click",
        "gvar",
        "lsqfit",
        "h5py",
    ],  # Dependencies
    python_requires=">=3.8",       # Minimum Python version
)
