from setuptools import setup, find_packages

setup(
    name="xerces_training",
    version="0.0.1",
    description="Training tools for Xerces",
    author="Bryan Goggin",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # add runtime deps here, e.g. 'pyfastchess'
    ],
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
