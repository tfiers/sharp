# Some tests for _sharp_

### Unit tests
Install test runner:
```
pip install pytest
```

Run tests:
```
pytest
```


### System test

Go to the correct working directory (containing a `luigi.toml` file):
```
cd integration/
```

Run system test:
```
python -m sharp --local-scheduler --clear-all
```
