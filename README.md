# 🔍 Glimpse

**Function-level logging and tracing for Python applications.**

Glimpse automatically captures your application's execution flow, giving you detailed insights into function calls, performance, and errors. Think of it as a microscope for your code - see exactly what's happening, when, and why.

Built by developers, for developers who need to understand their code better.

---

## ✨ Why Glimpse?

**Get visibility into your application without changing your code.** Glimpse automatically traces function execution, captures arguments and return values, and logs everything to your preferred storage backend.

Perfect for debugging complex applications, performance analysis, and understanding code flow in production systems.

**Key Benefits:**

- 🚀 **Zero-code tracing** - Just run `tracer.run()` and see everything
- 🎯 **Smart filtering** - Only trace what matters with flexible policies
- 📊 **Multiple backends** - Store traces in JSON, JSONL, SQLite, or build your own
- ⚡ **Performance optimized** - Minimal overhead, designed for production use
- 🔧 **Developer friendly** - Simple setup, clear output, easy integration

---

## 🚀 Quick Start

### Installation

```bash
pip install glimpse-py
```

### 30-Second Example

```python
from glimpse.tracer import Tracer
from glimpse.config import Config
from glimpse.policy import TracingPolicy

# Configure what to trace
config = Config(dest="jsonl", level="INFO")
policy = TracingPolicy(exact_modules=["myapp"], package_trees=["myapp.services"])

# Start automatic tracing
tracer = Tracer(config, policy=policy)
tracer.run()

# Your code runs here - everything gets traced automatically
def calculate_total(items):
    return sum(item.price for item in items)

result = calculate_total(shopping_cart)

tracer.stop()
```

**That's it!** Every function call matching your policy is now logged with timing, arguments, and results.

---

## 🛠️ Core Features

### Automatic Function Tracing

**Trace all function calls without decorators.** Just call `tracer.run()` and Glimpse automatically captures execution flow based on your policy.

```python
# Traces everything matching your policy
with tracer:
    your_application_code()
```

### Manual Decorators

**Fine-grained control for specific functions.** Use decorators when you want explicit tracing of important functions.

```python
@tracer.trace_function
def critical_function(data):
    # This function will always be traced
    return process_data(data)
```

### Multiple Storage Backends

**Store traces wherever you need them.** Built-in support for JSON, JSONL, and SQLite.

```python
from glimpse.config import Config

# JSON Lines (great for log analysis)
config = Config(dest="jsonl", params={"log_path": "/var/logs/traces.jsonl"})

# SQLite (great for local development and analysis)
config = Config(dest="sqlite", params={"db_path": "traces.db"})

# Multiple destinations simultaneously
config = Config(dest=["jsonl", "sqlite"])
```

### Smart Policy System

**Control exactly what gets traced.** Use JSON policies to include/exclude packages with powerful pattern matching.

```python
from glimpse.policy import TracingPolicy

# Programmatic policy creation
policy = TracingPolicy(
    exact_modules=["myapp.utils", "requests"],  # Exact module matches only
    package_trees=["myapp.services"],           # Package + all submodules
    trace_depth=10                              # Prevent infinite recursion
)

# Load from JSON file
policy = TracingPolicy.load("my-custom-policy.json")

# Auto-discover policy file
policy = TracingPolicy.load()  # Finds closest glimpse-policy.json
```

### Context Manager Support

**Clean, Pythonic usage.** Automatically start and stop tracing with context managers.

```python
with Tracer(config, policy=policy):
    # Tracing active only in this block
    run_my_application()
# Tracing automatically stops and cleans up
```

---

## ⚙️ Configuration

### Basic Configuration

**Configure storage, log levels, and output format.** Settings can be provided via constructor or environment variables.

```python
from glimpse.config import Config

# Constructor approach
config = Config(
    dest="jsonl",              # Storage backend: jsonl, json, sqlite
    level="INFO",              # Log level
    enable_trace_id=True,      # Enable trace correlation
    params={"log_path": "/var/logs/traces.jsonl"},
    max_field_length=512       # Truncate long values
)

# Environment variable approach
# GLIMPSE_DEST=jsonl
# GLIMPSE_LEVEL=DEBUG
# GLIMPSE_LOG_PATH=/var/logs/traces.jsonl
config = Config()  # Automatically loads from environment
```

### Environment Variables

All configuration can be controlled via environment variables:

| Variable           | Description                       | Default         |
| ------------------ | --------------------------------- | --------------- |
| `GLIMPSE_DEST`     | Storage backend (jsonl, sqlite)  | `jsonl`         |
| `GLIMPSE_LEVEL`    | Log level (INFO, DEBUG, ERROR)   | `INFO`          |
| `GLIMPSE_LOG_PATH` | Path to log file                  | `glimpse.jsonl` |
| `GLIMPSE_TRACE_ID` | Enable trace correlation          | `false`         |

---

## 📋 Policy System

### Policy Files

**Control tracing behavior with JSON policies.** Create policy files to define what gets traced.

```json
{
    "version": "1.0",
    "name": "my_policy",
    "exact_modules": ["mylib", "requests"],
    "package_trees": ["myapp", "services"],
    "trace_depth": 5
}
```

### Policy Types

**Two distinct filtering approaches for maximum flexibility:**

- **`exact_modules`**: Match only the exact module names specified (no submodules)
- **`package_trees`**: Match the package and ALL its submodules

```python
from glimpse.policy import TracingPolicy

policy = TracingPolicy(
    exact_modules=["requests", "urllib3"],        # Only these exact modules
    package_trees=["myapp", "services"],         # These packages + submodules
    trace_depth=10
)
```

### Wildcard Patterns

**Use powerful patterns to match modules.** Support for shell-style wildcards:

```json
{
    "exact_modules": [
        "api_v*",          // Matches: api_v1, api_v2, api_version_new
        "test_*_utils",    // Matches: test_unit_utils, test_integration_utils
        "config[12]"       // Matches: config1, config2
    ],
    "package_trees": [
        "myapp.*",         // Matches: myapp.services, myapp.utils, etc.
        "*.models"         // Matches: core.models, user.models, etc.
    ]
}
```

### Policy Discovery

**Automatic policy discovery.** Glimpse walks up your directory tree to find the closest policy file.

```python
# Auto-discover from caller location
policy = TracingPolicy.load()

# Explicit path (must be named 'glimpse-policy.json')
policy = TracingPolicy.load("/path/to/glimpse-policy.json")
```

```
myproject/
├── glimpse-policy.json     # Found and used automatically
├── src/
│   └── myapp/
│       └── main.py         # TracingPolicy.load() called here
└── tests/
```

---

## 📊 Output Examples

### JSONL Output

**Each function call produces a structured log entry.** Easy to analyze with standard tools like `jq`, ELK stack, or pandas.

```json
{"entry_id": "1234-a1b2c3d4-000001", "call_id": "abc123def456", "trace_id": "xyz789uvw012", "level": "INFO", "function": "myapp.services.get_user", "args": "get_user(user_id=123)", "stage": "START", "timestamp": "2023-12-01 10:30:15.123456"}
{"entry_id": "1234-a1b2c3d4-000002", "call_id": "abc123def456", "trace_id": "xyz789uvw012", "level": "INFO", "function": "myapp.services.get_user", "args": "get_user", "stage": "END", "result": "User(id=123, name='John')", "timestamp": "2023-12-01 10:30:15.145223", "duration_ms": "21.767"}
```

### SQLite Output

**Structured database storage for complex analysis.** Query your traces with SQL for powerful insights.

```sql
-- Find slow functions
SELECT function, AVG(duration_ms) as avg_duration 
FROM trace_entries 
WHERE stage = 'END' 
GROUP BY function 
ORDER BY avg_duration DESC;

-- Find error patterns
SELECT function, COUNT(*) as error_count 
FROM trace_entries 
WHERE stage = 'EXCEPTION' 
GROUP BY function;

-- Trace correlation by call_id
SELECT * FROM trace_entries 
WHERE call_id = 'abc123def456' 
ORDER BY timestamp;
```

### Trace Analysis

**Easily analyze execution patterns:**

```bash
# Find slow functions (JSONL)
cat traces.jsonl | jq 'select(.duration_ms > 100)'

# Count function calls
cat traces.jsonl | jq '.function' | sort | uniq -c

# Find errors
cat traces.jsonl | jq 'select(.stage == "EXCEPTION")'

# Trace specific call flow
cat traces.jsonl | jq 'select(.call_id == "abc123def456")'
```

---

## 🏗️ Architecture

### High-Level Design

**Modular architecture designed for extensibility.** Each component has a single responsibility and can be extended or replaced.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Tracer    │───▶│    Policy    │───▶│   Writer    │
│             │    │              │    │             │
│ • Manual    │    │ • Exact Mods │    │ • JSON      │
│ • Automatic │    │ • Pkg Trees  │    │ • JSONL     │
│ • Context   │    │ • Wildcards  │    │ • SQLite    │
└─────────────┘    └──────────────┘    └─────────────┘
```

### Component Overview

- **Tracer**: Core tracing engine using `sys.settrace()`
- **TracingPolicy**: Smart filtering with exact modules vs package trees
- **Config**: Environment-aware configuration management
- **Writers**: Pluggable storage backends (JSON, JSONL, SQLite)
- **LogEntry**: Structured trace data model with correlation IDs
- **IDGenerator**: Thread-safe ID generation for tracing correlation

### Performance Characteristics

**Designed for production use.** Optimized hot paths, minimal overhead, and efficient pattern matching.

- ⚡ **Efficient pattern matching** using tries and sets
- 🎯 **Smart filtering** to avoid tracing unnecessary code
- 📊 **Configurable depth limits** to prevent runaway traces
- 🔧 **Self-filtering** to avoid tracing the tracer itself
- 🧵 **Thread-safe** ID generation and SQLite operations

---

## 🚀 Upcoming Features

### Enhanced Policy System

**More sophisticated filtering and configuration options.** Coming soon: regex patterns, performance-based filtering, and dynamic policies.

### Distributed System Support

**Trace across multiple services and processes.** Coming soon: trace correlation, request IDs, and cross-service visibility.

### Additional Storage Backends

**More storage options.** Planned: MongoDB, Elasticsearch, and cloud storage integrations.

### Custom Writers

**Build your own storage backends.** Simple interface for integrating with any storage system.

```python
# Future: Custom storage backends
from glimpse.writers.base import BaseWriter

class ElasticsearchWriter(BaseWriter):
    def write(self, entry):
        # Your custom logic here
        pass
```

---

## 🤝 Contributing

Found a bug? Have a feature request? Want to contribute code?

- 📝 [Open an issue](https://github.com/PxPatel/glimpse/issues)
- 🔀 [Submit a pull request](https://github.com/PxPatel/glimpse/pulls)
- 💬 [Start a discussion](https://github.com/PxPatel/glimpse/discussions)

**Built by developers, for developers.** Your feedback and contributions make Glimpse better for everyone.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**⭐ Like Glimpse? Give us a star and help other developers discover it!**