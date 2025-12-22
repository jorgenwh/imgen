.PHONY: clean install

clean:
	rm -rf outputs/* models/*

install:
	uv sync
	uv pip install -e .

