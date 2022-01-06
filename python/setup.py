from setuptools import setup

setup(
    name='slitless',
    version="1",
    packages=['slitless'],
    author="Ulas Kamaci",
    author_email="ukamaci2@illinois.edu",
    description="Slitless Spectral Imaging Reconstruction Code",
    long_description=open('README.md').read(),
    license="GPLv3",
    keywords="slitless spectral imaging reconstruction",
    url="https://github.com/ukamaci/slitless",
    install_requires=[
        "matplotlib",
        "numpy",
        "scikit-image",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
