from setuptools import setup, find_packages


setup(
    name="dain_package",
    version="0.0.1",
    description="package version of dain",
    url="https://github.com/realiti4/dain",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=["numpy", "pandas"],
)
