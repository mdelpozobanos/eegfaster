.PHONY: clean-pyc clean-build docs clean

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "flake8 - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "dist - package"
	@echo "install - install the package to the active Python's site-packages"
	@echo "dev-doc - generate Sphinx HTML documentation with private methods, including API docs"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

flake8:
	flake8 eegfaster tests

test:
	python setup.py test

test-all:
	tox

coverage:
	coverage run --source eegfaster setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/eegfaster.rst
	rm -f docs/modules.rst
	sphinx-apidoc -fMo docs/source eegfaster
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/build/html/index.html

release: clean
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean
	python setup.py install

dev-git:
	git flow init -d
	git add .
	git commit -m 'Initial commit'

dev-git-hub:
	git remote add origin git+ssh://git@github.com/mdelpozobanos/eegfaster
	git push origin develop