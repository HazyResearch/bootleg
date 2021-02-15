dev:
	pip install -r requirements-dev.txt
	python3 setup.py develop
	pre-commit install

test: dev check docs
	pip install -e .
	pytest test

format:
	isort --atomic bootleg/ test/
	black bootleg/ test/
	docformatter --in-place --recursive bootleg test

check:
	isort -c -rc bootleg/ test/
	black bootleg/ test/ --check
  	# flake8 bootleg/ test/

docs:
	sphinx-build -b html docs/source/ docs/build/html/
	# sphinx-apidoc -o docs/source/apidocs/ bootleg

docs-check:
	sphinx-build -b html docs/source/ docs/build/html/ -W

livedocs:
	sphinx-autobuild -b html docs/source/ docs/build/html/

clean:
	pip uninstall -y bootleg
	rm -rf src/bootleg.egg-info
	rm -rf _build/

.PHONY: dev test clean check docs
