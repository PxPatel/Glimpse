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
- 📊 **Multiple backends** - Store traces in JSON, SQLite, MongoDB, or build your own
- ⚡ **Performance optimized** - Minimal overhead, designed for production use
- 🔧 **Developer friendly** - Simple setup, clear output, easy integration

---

## 🚀 Quick Start

### Installation
```bash
pip install glimpse  # Coming soon to PyPI
# For now:
git clone https://github.com/your-username/glimpse
cd glimpse
pip install -e .
```

### 30-Second Example
```python
from glimpse import Tracer, Config, TracingPolicy

# Configure what to trace
config = Config(dest="jsonl", level="INFO")
policy = TracingPolicy(included_packages=["myapp"])

# Start automatic tracing
tracer = Tracer(config, policy=policy)
tracer.run()

# Your code runs here - everything gets traced automatically
def calculate_total(items):
    return sum(item.price for item in items)

result = calculate_total(shopping_cart)

tracer.stop()
```

**That's it!** Every function call in `myapp` is now logged with timing, arguments, and results.

---

## 🛠️ Core Features

### Automatic Function Tracing
**Trace all function calls without decorators.** Just call `tracer.run()` and Glimpse automatically captures execution flow based on your policy.

```python
# Traces everything in your application matching the policy
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
**Store traces wherever you need them.** Built-in support for JSON, JSONL, SQLite, and MongoDB (coming soon).

```python
# JSON Lines (great for log analysis)
config = Config(dest="jsonl", params={"log_path": "/var/logs/traces.jsonl"})

# SQLite (great for local development)  
config = Config(dest="sqlite", params={"db_path": "traces.db"})
```

### Smart Policy System
**Control exactly what gets traced.** Use JSON policies to include/exclude packages with powerful wildcard patterns.

```python
policy = TracingPolicy(
    included_packages=["myapp", "requests"],  # Exact matches
    project_root_packages=["src*"],           # Wildcard patterns
    trace_depth=10                            # Prevent infinite recursion
)
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
from glimpse import Config

# Constructor approach
config = Config(
    dest="jsonl",              # Storage backend
    level="INFO",              # Log level
    max_field_length=512,      # Truncate long values
    params={"log_path": "/var/logs/traces.jsonl"}
)

# Environment variable approach
# GLIMPSE_DEST=jsonl
# GLIMPSE_LEVEL=DEBUG  
# GLIMPSE_LOG_PATH=/var/logs/traces.jsonl
config = Config()  # Automatically loads from environment
```

### Environment Variables
All configuration can be controlled via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GLIMPSE_DEST` | Storage backend (jsonl, sqlite, mongo) | `jsonl` |
| `GLIMPSE_LEVEL` | Log level (INFO, DEBUG, ERROR) | `INFO` |
| `GLIMPSE_LOG_PATH` | Path to log file | `glimpse.jsonl` |
| `GLIMPSE_TRACE_ID` | Enable trace correlation | `false` |

---

## 📋 Policy System

### Policy Files
**Control tracing behavior with JSON policies.** Glimpse automatically discovers `glimpse-policy.json` files in your project.

```json
{
  "version": "1.0",
  "included_packages": ["myapp", "requests"],
  "project_root_packages": ["src"],
  "trace_depth": 5,
  "auto_trace_subpackages": true
}
```

### Wildcard Patterns
**Use powerful patterns to match packages.** Support for shell-style wildcards and character classes.

```json
{
  "included_packages": [
    "myapp",           // Exact match: myapp and myapp.submodule
    "api_v*",          // Wildcard: api_v1, api_v2, api_version_new
    "test_*_utils",    // Complex: test_unit_utils, test_integration_utils
    "*.models",        // Suffix: myapp.models, core.user.models
    "config[12]"       // Character class: config1, config2
  ]
}
```

### Policy Discovery
**Automatic policy discovery.** Glimpse walks up your directory tree to find the closest `glimpse-policy.json` file.

```
myproject/
├── glimpse-policy.json     # Found and used automatically
├── src/
│   └── myapp/
│       └── main.py         # Tracer initialized here
└── tests/
```

---

## 📊 Output Examples

### JSONL Output
**Each function call produces a structured log entry.** Easy to analyze with standard tools like `jq`, ELK stack, or pandas.

```json
{"entry_id": 1, "level": "INFO", "function": "myapp.services.get_user", "args": "get_user(user_id=123)", "stage": "START", "timestamp": "2023-12-01 10:30:15.123456"}
{"entry_id": 2, "level": "INFO", "function": "myapp.services.get_user", "args": "get_user", "stage": "END", "result": "User(id=123, name='John')", "timestamp": "2023-12-01 10:30:15.145223", "duration_ms": "21.767"}
```

### Trace Analysis
**Easily analyze execution patterns:**

```bash
# Find slow functions
cat traces.jsonl | jq 'select(.duration_ms > 100)'

# Count function calls
cat traces.jsonl | jq '.function' | sort | uniq -c

# Find errors
cat traces.jsonl | jq 'select(.stage == "EXCEPTION")'
```

---

## 🏗️ Architecture

### High-Level Design
**Modular architecture designed for extensibility.** Each component has a single responsibility and can be extended or replaced.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Tracer    │───▶│    Policy    │───▶│   Writer    │
│             │    │              │    │             │
│ • Manual    │    │ • Wildcards  │    │ • JSON      │
│ • Automatic │    │ • Discovery  │    │ • SQLite    │
│ • Context   │    │ • Filtering  │    │ • MongoDB   │
└─────────────┘    └──────────────┘    └─────────────┘
```

### Component Overview
- **Tracer**: Core tracing engine using `sys.settrace()`
- **Policy**: Smart filtering system with wildcard support
- **Config**: Environment-aware configuration management  
- **Writers**: Pluggable storage backends
- **LogEntry**: Structured trace data model

### Performance Characteristics
**Designed for production use.** Optimized hot paths, minimal overhead, and efficient pattern matching.

- ⚡ **O(1) exact pattern matching** using sets
- 🎯 **Smart filtering** to avoid tracing unnecessary code  
- 📊 **Configurable depth limits** to prevent runaway traces
- 🔧 **Self-filtering** to avoid tracing the tracer itself

---

## 🚀 Upcoming Features

### Distributed System Support
**Trace across multiple services and processes.** Coming soon: trace correlation, request IDs, and cross-service visibility.

```python
# Future: Distributed tracing
tracer = Tracer(config, policy=policy, correlation_id="req-123")
```

### Multi-Policy Support
**Different policies for different parts of your application.** Perfect for microservices and complex applications.

<!-- ```python
# Future: Service-specific policies
tracer = Tracer(config, policies={
    "user-service": user_policy,
    "payment-service": payment_policy
})
``` -->

### Custom Writers
**Build your own storage backends.** Simple interface for integrating with any storage system.

```python
# Future: Custom storage backends
class ElasticsearchWriter(BaseWriter):
    def write(self, entry):
        # Your custom logic here
        pass
```

---

## 🤝 Contributing

Found a bug? Have a feature request? Want to contribute code?

- 📝 [Open an issue](https://github.com/your-username/glimpse/issues)
- 🔀 [Submit a pull request](https://github.com/your-username/glimpse/pulls)
- 💬 [Start a discussion](https://github.com/your-username/glimpse/discussions)

**Built by developers, for developers.** Your feedback and contributions make Glimpse better for everyone.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**⭐ Like Glimpse? Give us a star and help other developers discover it!**