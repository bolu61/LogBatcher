# LogBatcher

LogBatcher is a log parsing and clustering tool that groups similar log messages into templates. It's a fork of the [original LogBatcher](https://github.com/logintelligence/logbatcher) with an improved interface. Installation from source only.

## Usage

```python
from logbatcher import LogBatcher

# Initialize the parser
parser = LogBatcher()

# Parse logs
logs = [
    "User login failed for user admin",
    "User login failed for user guest",
    "User login successful for user admin"
]

templates = parser.parse(logs)
print(templates)
```

## Core Components

- `LogBatcher`: Main class for parsing logs
- `Cluster`: Represents a group of similar logs
- `ParsingCache`: Handles template caching
- `matching`: Log matching and pruning logic
- `cluster`: Clustering algorithms
- `postprocess`: Template post-processing

## License

MIT License - see LICENSE file for details.
