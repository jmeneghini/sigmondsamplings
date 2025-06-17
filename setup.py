#!/usr/bin/env python3
"""
Setup script for SigmondSamplings package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A Python package for handling Sigmond samplings files"

# Read version from __init__.py
def get_version():
    init_path = os.path.join('SigmondSamplings', '__init__.py')
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

setup(
    name='SigmondSamplings',
    version=get_version(),
    author='John Meneghini',
    author_email='jmeneghi@andrew.cmu.edu',
    description='A Python package for handling Sigmond samplings files',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/jmeneghini/SigmondSamplings',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ],
    },
    include_package_data=True,
    package_data={
        'SigmondSamplings': ['*.txt', '*.md'],
    },
    zip_safe=False,
    keywords='lattice QCD, Monte Carlo, bootstrap, jackknife, statistics',
    project_urls={
        'Source': 'https://github.com/jmeneghini/SigmondSamplings',
    },
) 