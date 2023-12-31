PROJECT=rusty-data-profiler
PYTHON_VERSION=3.10.10
SOURCE_OBJECTS=rusty_data_profiler


format.black:
	poetry run black ${SOURCE_OBJECTS}
format.ruff:
	poetry run ruff check --silent --fix --exit-zero ${SOURCE_OBJECTS}
format: format.ruff format.black

lints.format_check:
	poetry run black --check ${SOURCE_OBJECTS}
lints.ruff:
	poetry run ruff check ${SOURCE_OBJECTS}
lints.mypy:
	poetry run mypy ${SOURCE_OBJECTS}
lints: lints.ruff lints.mypy
lints.ci: lints.format_check lints.ruff lints.mypy

setup.project:
	poetry install --all-extras --with dev,dev-lints

test.unit:
	poetry run pytest \
		--cov \
		--cov-fail-under=0 \
		--cov-report xml:./coverage.xml \
		--cov-report term \
		--junitxml=./results.xml

poetry.pre.patch:
	poetry version prepatch

poetry.sub.pre.tag:
	$(eval VER = $(shell grep "^version =" pyproject.toml | tr -d '"' | sed "s/^version = //"))
	$(eval TS = $(shell date +%s))
	$(eval REL_CANDIDATE = $(subst a0,rc.$(TS),$(VER)))
	@sed -i "s/$(VER)/$(REL_CANDIDATE)/" pyproject.toml

prep.pre.patch: poetry.pre.patch poetry.sub.pre.tag

publish:
	poetry publish --repository  --build

publish.docs:
	cd docs && poetry run mkdocs gh-deploy --force