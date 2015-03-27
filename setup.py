#!/usr/bin/env python

import io
import re
from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys


def find_version(*file_paths):
    """
    Finds the *__version__* of a package by reading it from __init__.py
    """
    version_file = read(*file_paths)
    version_match = re.search('^__version__ = [\'](.+)[\']',
                              version_file, re.M)

    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read(*filenames, **kwargs):
    """Reads files and return their content in a single string"""
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


# Special code to allow "python setup.py test" to work properly
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

# Common requirements
install_requires = [
    'numpy>=1.0, <2',
    'scipy>=0.0, <1'
]
tests_require = [
    'pytest>=2.0, <3',
    'coverage>=3.0, <4.0',
    'flake8>=2.0, <3.0',
    'wheel>=0.0, <1.0',
    'tox>=1.0, <2.0'
]
# Conditional requirements
if sys.version > '3':
    pass
else:
    pass

setup(
    name='eegfaster',
    version=find_version('eegfaster/__init__.py'),
    description='FASTER method for EEG artifact rejection',
    long_description=read('README.rst') + read('HISTORY.rst'),
    author='Marcos DelPozo-Banos',
    author_email='mdelpozobanos@gmail.com',
    url='https://github.com/mdelpozobanos/eegfaster',
    packages=[
        'eegfaster',
    ],
    package_dir={'eegfaster': 'eegfaster'},
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    cmdclass={'test': PyTest},
    license='MIT',
    zip_safe=False,
    keywords='eegfaster',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python'
    ],
)