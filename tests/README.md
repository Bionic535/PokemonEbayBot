## How to run tests

The tests use [pytest](https://docs.pytest.org/)

**Requirements:** A completed `.env` file

From the project root run

For all tests
```sh
pytest
```

Or

```sh
pytest -v    #verbose output
pytest -m integration #only integration tests
pytest tests/test_llm_call_flow.py # one file
pytest -k charizard                # filter by case name
```


