How to Run Tests and Measure Coverage

Install test deps
- `pip install -r model_server/requirements-dev.txt`

Run all tests
- `pytest`

Coverage
- Configured via `pytest.ini` to report coverage for `model_server`.
- View missing lines in terminal with `--cov-report=term-missing`.

Notes
- Tests avoid importing `llama-cpp-python` by monkeypatching `get_llama`.
- For streaming tests, we use FastAPI’s `TestClient.stream`.
