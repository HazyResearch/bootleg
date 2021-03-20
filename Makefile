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
	isort -c bootleg/ test/
	black bootleg/ test/ --check
	flake8 bootleg/ test/

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

prune:
	@bash -c "git fetch -p";
	@bash -c "for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do git branch -d $branch; done";

.PHONY: dev test clean check docs prune
