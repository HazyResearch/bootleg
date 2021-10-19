dev:
	pip install -e .[dev]
	pre-commit install

test: dev check docs
	pip install -e .
	pytest tests

format:
	isort --atomic bootleg/ tests/
	black bootleg/ tests/
	# docformatter --in-place --recursive bootleg tests

check:
	isort -c bootleg/ tests/
	black bootleg/ tests/ --check
	flake8 bootleg/ tests/

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
	rm -rf build/ dist/

prune:
	@bash -c "git fetch -p";
	@bash -c "for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do git branch -d $branch; done";

.PHONY: dev test clean check docs prune
