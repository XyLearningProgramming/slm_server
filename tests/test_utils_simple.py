"""Simple tests for slm_server.utils module without complex tracing setup."""

from unittest.mock import Mock

import pytest

from slm_server.model import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    Usage,
    ChatCompletionChoice,
    ChatCompletionStreamChoice,
    DeltaMessage,
)
from slm_server.utils import (
    ATTR_CHUNK_COUNT,
    EVENT_ATTR_CHUNK_CONTENT_SIZE,
    EVENT_ATTR_CHUNK_SIZE,
    EVENT_CHUNK_GENERATED,
    METRIC_AVG_CHUNK_CONTENT_SIZE,
    METRIC_AVG_CHUNK_SIZE,
    METRIC_CHUNK_DELAY,
    METRIC_CHUNKS_WITH_CONTENT,
    METRIC_EMPTY_CHUNKS,
    METRIC_FIRST_TOKEN_DELAY,
    METRIC_MAX_CHUNK_SIZE,
    METRIC_MIN_CHUNK_SIZE,
    METRIC_TOKENS_PER_SECOND,
    METRIC_TOTAL_DURATION,
    METRIC_TOTAL_TOKENS_PER_SECOND,
    calculate_performance_metrics,
    set_atrribute_response,
    set_atrribute_response_stream,
)
from slm_server.utils.metrics import _calculate_chunk_metrics_from_events


class TestSetAttributeResponse:
    """Tests for set_atrribute_response function."""
    
    def test_sets_attributes_correctly(self):
        """Test that response attributes are set correctly on span."""
        mock_span = Mock()
        
        # Create response with usage and content
        response = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello world!"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        
        set_atrribute_response(mock_span, response)
        
        # Verify attributes were set
        mock_span.set_attribute.assert_any_call("slm.usage.prompt_tokens", 10)
        mock_span.set_attribute.assert_any_call("slm.usage.completion_tokens", 5)
        mock_span.set_attribute.assert_any_call("slm.usage.total_tokens", 15)
        mock_span.set_attribute.assert_any_call("slm.output.content_length", 12)  # len("Hello world!")
    
    def test_handles_empty_content(self):
        """Test handling of empty content."""
        mock_span = Mock()
        
        # Create response with empty content
        response = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=""),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
        )
        
        set_atrribute_response(mock_span, response)
        
        # Should set content length to 0
        mock_span.set_attribute.assert_any_call("slm.output.content_length", 0)


class TestSetAttributeResponseStream:
    """Tests for set_atrribute_response_stream function."""
    
    def test_records_chunk_event(self):
        """Test that chunk events are recorded correctly."""
        mock_span = Mock()
        mock_span.attributes = Mock()
        mock_span.attributes.get = Mock(return_value=0)
        
        # Create streaming response chunk
        response = ChatCompletionStreamResponse(
            model="test-model",
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(content="Hello"),
                    finish_reason=None
                )
            ]
        )
        
        set_atrribute_response_stream(mock_span, response)
        
        # Verify event was added
        mock_span.add_event.assert_called_once()
        event_name, event_attrs = mock_span.add_event.call_args[0]
        
        assert event_name == EVENT_CHUNK_GENERATED
        assert event_attrs[EVENT_ATTR_CHUNK_CONTENT_SIZE] == 5
        assert event_attrs[EVENT_ATTR_CHUNK_SIZE] > 0  # JSON size
    
    def test_ignores_empty_chunks(self):
        """Test that empty chunks don't increment counters."""
        mock_span = Mock()
        mock_span.attributes = Mock()
        mock_span.attributes.get = Mock(return_value=0)
        
        response = ChatCompletionStreamResponse(
            model="test-model",
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=""),
                    finish_reason=None
                )
            ]
        )
        
        set_atrribute_response_stream(mock_span, response)
        
        # Should add event but not set attributes for empty content
        mock_span.add_event.assert_called_once()
        mock_span.set_attribute.assert_not_called()


class TestCalculateChunkMetricsFromEvents:
    """Tests for _calculate_chunk_metrics_from_events function."""
    
    def test_calculates_metrics_correctly(self):
        """Test chunk metrics calculation from events."""
        # Create mock events - content sizes represent different chunks
        events = []
        content_sizes = [5, 0, 5, 1]  # "Hello", "", "world", "!"
        for i, content_size in enumerate(content_sizes):
            event = Mock()
            event.name = EVENT_CHUNK_GENERATED
            event.attributes = {
                EVENT_ATTR_CHUNK_SIZE: 100 + i * 10,  # Varying sizes
                EVENT_ATTR_CHUNK_CONTENT_SIZE: content_size,
            }
            events.append(event)
        
        metrics = _calculate_chunk_metrics_from_events(events)
        
        # Verify calculations
        assert metrics[METRIC_AVG_CHUNK_SIZE] == (100 + 110 + 120 + 130) / 4  # 115
        assert metrics[METRIC_MAX_CHUNK_SIZE] == 130
        assert metrics[METRIC_MIN_CHUNK_SIZE] == 100
        assert metrics[METRIC_AVG_CHUNK_CONTENT_SIZE] == (5 + 0 + 5 + 1) / 4  # 2.75
        assert metrics[METRIC_CHUNKS_WITH_CONTENT] == 3  # non-zero content sizes
        assert metrics[METRIC_EMPTY_CHUNKS] == 1  # zero content size
    
    def test_handles_empty_events(self):
        """Test handling of empty events list."""
        metrics = _calculate_chunk_metrics_from_events([])
        assert metrics == {}
    
    def test_filters_non_chunk_events(self):
        """Test that only chunk events are processed."""
        event1 = Mock()
        event1.name = "other_event"
        event1.attributes = {}
        
        event2 = Mock()
        event2.name = EVENT_CHUNK_GENERATED
        event2.attributes = {
            EVENT_ATTR_CHUNK_SIZE: 100,
            EVENT_ATTR_CHUNK_CONTENT_SIZE: 5,
        }
        
        events = [event1, event2]
        
        metrics = _calculate_chunk_metrics_from_events(events)
        
        # Should only process one chunk event with content
        assert metrics[METRIC_CHUNKS_WITH_CONTENT] == 1
        assert metrics[METRIC_EMPTY_CHUNKS] == 0


class TestCalculatePerformanceMetrics:
    """Tests for calculate_performance_metrics function."""
    
    def test_calculates_basic_metrics(self):
        """Test basic performance metrics calculation."""
        mock_span = Mock()
        mock_span.start_time = 1000_000_000  # 1 second in nanoseconds
        mock_span.end_time = 2000_000_000    # 2 seconds in nanoseconds
        mock_span.attributes = {
            "slm.usage.total_tokens": 100,
            "slm.usage.completion_tokens": 50,
            "slm.streaming": False,
        }
        mock_span.events = []
        
        metrics = calculate_performance_metrics(mock_span)
        
        assert metrics[METRIC_TOTAL_DURATION] == 1000.0  # 1 second = 1000ms
        assert metrics[METRIC_TOKENS_PER_SECOND] == 50.0  # 50 tokens / 1 second
        assert metrics[METRIC_TOTAL_TOKENS_PER_SECOND] == 100.0  # 100 tokens / 1 second
    
    def test_streaming_metrics_with_events(self):
        """Test streaming-specific metrics calculation."""
        mock_span = Mock()
        mock_span.start_time = 1000_000_000
        mock_span.end_time = 2000_000_000
        mock_span.attributes = {
            "slm.streaming": True,
            "slm.output.chunk_count": 3,
            "slm.usage.total_tokens": 60,
            "slm.usage.completion_tokens": 30,
        }
        
        # Mock first chunk event (first event is treated as first token)
        first_event = Mock()
        first_event.name = EVENT_CHUNK_GENERATED
        first_event.timestamp = 1100_000_000  # 100ms after start
        first_event.attributes = {EVENT_ATTR_CHUNK_CONTENT_SIZE: 5}
        
        # Mock other chunk events
        other_events = []
        content_sizes = [0, 5]  # "", "world"
        for content_size in content_sizes:
            event = Mock()
            event.name = EVENT_CHUNK_GENERATED
            event.attributes = {
                EVENT_ATTR_CHUNK_SIZE: 100,
                EVENT_ATTR_CHUNK_CONTENT_SIZE: content_size,
            }
            other_events.append(event)
        
        mock_span.events = [first_event] + other_events
        
        metrics = calculate_performance_metrics(mock_span)
        
        assert metrics[METRIC_CHUNK_DELAY] == 1000.0 / 3  # 1000ms / 3 chunks
        assert metrics[METRIC_FIRST_TOKEN_DELAY] == 100.0  # 100ms to first content
        assert METRIC_CHUNKS_WITH_CONTENT in metrics
        assert METRIC_EMPTY_CHUNKS in metrics
    
    def test_handles_missing_timestamps(self):
        """Test handling of spans without proper timestamps."""
        mock_span = Mock()
        mock_span.start_time = None
        mock_span.end_time = None
        
        metrics = calculate_performance_metrics(mock_span)
        assert metrics == {}


def test_streaming_call_integration():
    """Integration test for streaming call simulation."""
    # Test the key functions work together
    mock_span = Mock()
    mock_span.attributes = Mock()
    
    # Setup attributes tracker
    attributes = {}
    def mock_get(key, default=None):
        return attributes.get(key, default)
    
    def mock_set(key, value):
        attributes[key] = value
        
    mock_span.attributes.get = mock_get
    mock_span.set_attribute = mock_set
    
    # Simulate chunks
    chunks = [
        ChatCompletionStreamResponse(
            model="test-model",
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(content="Hello"),
                finish_reason=None
            )]
        ),
        ChatCompletionStreamResponse(
            model="test-model",
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(content=" world!"),
                finish_reason=None
            )]
        ),
        ChatCompletionStreamResponse(
            model="test-model",
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(content=""),
                finish_reason="stop"
            )]
        )
    ]
    
    # Process chunks
    for chunk in chunks:
        set_atrribute_response_stream(mock_span, chunk)
    
    # Verify final state
    assert attributes["slm.usage.completion_tokens"] == 2  # "Hello" + " world!"
    assert attributes["slm.output.content_length"] == 12  # "Hello world!"
    assert attributes["slm.output.chunk_count"] == 2  # Only chunks with content
    
    # Verify events were recorded
    assert mock_span.add_event.call_count == 3  # All chunks including empty


def test_non_streaming_call_integration():
    """Integration test for non-streaming call simulation."""
    mock_span = Mock()
    
    # Create response
    response = ChatCompletionResponse(
        model="test-model",
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello world! How are you?"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=15, completion_tokens=20, total_tokens=35)
    )
    
    set_atrribute_response(mock_span, response)
    
    # Verify attributes were set
    mock_span.set_attribute.assert_any_call("slm.usage.prompt_tokens", 15)
    mock_span.set_attribute.assert_any_call("slm.usage.completion_tokens", 20)
    mock_span.set_attribute.assert_any_call("slm.usage.total_tokens", 35)
    mock_span.set_attribute.assert_any_call("slm.output.content_length", 25)  # length of response