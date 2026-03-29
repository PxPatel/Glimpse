"""Integration tests for end-to-end Jaeger export via LogWriter."""
import sys
from unittest.mock import MagicMock, patch

import pytest

from glimpse.span import Span
from glimpse.config import Config
from glimpse.writers.logwriter import LogWriter


@pytest.fixture
def sample_span():
    return Span(
        trace_id="abc123",
        span_id="def456",
        name="test-op",
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T00:00:01",
    )


def _make_logwriter_with_jaeger(params=None):
    """
    Build a LogWriter backed by JaegerWriter, with requests mocked out.

    Returns (logwriter, mock_requests) so callers can assert on post calls.
    """
    mock_requests = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = ""
    mock_requests.post.return_value = mock_response

    config = Config(dest="json", params=params)
    lw = LogWriter(config, writer_initiation=False)

    with patch.dict("sys.modules", {"requests": mock_requests}):
        jaeger_writer = lw._initialize_destination("jaeger")

    # Inject the mock so calls after construction are captured
    jaeger_writer._requests = mock_requests
    lw._writers.append(jaeger_writer)

    return lw, mock_requests


def test_logwriter_write_span_reaches_jaeger(sample_span):
    """write_span on LogWriter with jaeger dest calls requests.post once with resourceSpans."""
    lw, mock_requests = _make_logwriter_with_jaeger()
    lw.write_span(sample_span)

    assert mock_requests.post.call_count == 1
    payload = mock_requests.post.call_args[1]["json"]
    assert "resourceSpans" in payload


def test_jaeger_export_failure_does_not_raise(sample_span, capsys):
    """ConnectionError from requests.post does not propagate — error message appears on stderr."""
    lw, mock_requests = _make_logwriter_with_jaeger()
    mock_requests.post.side_effect = ConnectionError("connection refused")

    # Must not raise
    lw.write_span(sample_span)

    captured = capsys.readouterr()
    assert "Glimpse" in captured.err


def test_jaeger_extra_endpoint_config(sample_span):
    """Custom jaeger_endpoint param is forwarded to requests.post as the URL."""
    custom_url = "http://custom:4318/v1/traces"
    lw, mock_requests = _make_logwriter_with_jaeger(params={"jaeger_endpoint": custom_url})
    lw.write_span(sample_span)

    assert mock_requests.post.call_count == 1
    called_url = mock_requests.post.call_args[0][0]
    assert called_url == custom_url
