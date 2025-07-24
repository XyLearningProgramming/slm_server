"""Tests for slm_server.utils module."""

import time
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode, set_tracer_provider

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
    # EVENT_ATTR_CHUNK_CONTENT,
    EVENT_ATTR_CHUNK_CONTENT_SIZE,
    EVENT_ATTR_CHUNK_SIZE,
    # EVENT_ATTR_FINISH_REASON,
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
    SLMLoggingSpanProcessor,
    SLMMetricsSpanProcessor,
    calculate_performance_metrics,
    set_atrribute_response,
    set_atrribute_response_stream,
    slm_span,
)
from slm_server.utils.metrics import _calculate_chunk_metrics_from_events
from slm_server.utils.spans import tracer


@pytest.fixture
def setup_tracing():
    """Set up tracing with in-memory span exporter for testing."""
    # Create a tracer provider with in-memory exporter
    tracer_provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    span_processor = SimpleSpanProcessor(memory_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Don't override global tracer provider - use local one
    local_tracer = tracer_provider.get_tracer(__name__)
    
    yield memory_exporter, local_tracer
    
    # Clean up
    memory_exporter.clear()


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
        mock_span.attributes = {}
        
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
    
    def test_accumulates_tokens(self):
        """Test that tokens are accumulated correctly."""
        mock_span = Mock()
        def mock_get(key, default=None):
            attrs = {
                "slm.usage.completion_tokens": 2,
                "slm.output.content_length": 10,
                "slm.usage.prompt_tokens": 5,
                "slm.output.chunk_count": 1
            }
            return attrs.get(key, default)
        
        mock_span.attributes = Mock()
        mock_span.attributes.get = mock_get
        
        response = ChatCompletionStreamResponse(
            model="test-model",
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(content="world"),
                    finish_reason=None
                )
            ]
        )
        
        set_atrribute_response_stream(mock_span, response)
        
        # Verify attributes were updated
        mock_span.set_attribute.assert_any_call("slm.usage.completion_tokens", 3)
        mock_span.set_attribute.assert_any_call("slm.output.content_length", 15)  # 10 + 5
        mock_span.set_attribute.assert_any_call("slm.usage.total_tokens", 8)  # 5 + 3
        mock_span.set_attribute.assert_any_call("slm.output.chunk_count", 2)
    
    def test_ignores_empty_chunks(self):
        """Test that empty chunks don't increment counters."""
        mock_span = Mock()
        mock_span.attributes = {}
        
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
        # Create mock events
        events = []
        for i, content in enumerate(["Hello", "", "world", "!"]):
            event = Mock()
            event.name = EVENT_CHUNK_GENERATED
            event.attributes = {
                EVENT_ATTR_CHUNK_SIZE: 100 + i * 10,  # Varying sizes
                EVENT_ATTR_CHUNK_CONTENT_SIZE: len(content),
            }
            events.append(event)
        
        metrics = _calculate_chunk_metrics_from_events(events)
        
        # Verify calculations
        assert metrics[METRIC_AVG_CHUNK_SIZE] == (100 + 110 + 120 + 130) / 4  # 115
        assert metrics[METRIC_MAX_CHUNK_SIZE] == 130
        assert metrics[METRIC_MIN_CHUNK_SIZE] == 100
        assert metrics[METRIC_AVG_CHUNK_CONTENT_SIZE] == (5 + 0 + 5 + 1) / 4  # 2.75
        assert metrics[METRIC_CHUNKS_WITH_CONTENT] == 3  # "Hello", "world", "!"
        assert metrics[METRIC_EMPTY_CHUNKS] == 1  # ""
    
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
        
        # Should only process one chunk event
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
        
        # Mock first chunk event
        first_event = Mock()
        first_event.name = EVENT_CHUNK_GENERATED
        first_event.timestamp = 1100_000_000  # 100ms after start
        first_event.attributes = {EVENT_ATTR_CHUNK_CONTENT_SIZE: 5}
        
        # Mock other chunk events
        other_events = []
        for content in ["", "world"]:
            event = Mock()
            event.name = EVENT_CHUNK_GENERATED
            event.attributes = {
                EVENT_ATTR_CHUNK_SIZE: 100,
                EVENT_ATTR_CHUNK_CONTENT_SIZE: len(content),
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


class TestSlmSpan:
    """Tests for slm_span context manager."""
    
    def test_sets_initial_attributes(self, setup_tracing):
        """Test that initial attributes are set correctly."""
        memory_exporter, local_tracer = setup_tracing
        
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there")
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_span(request, is_streaming=True) as (span, messages):
                pass
        
        # Get the finished span
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        attrs = span.attributes
        
        assert attrs["slm.model"] == "llama-cpp"
        assert attrs["slm.streaming"] is True
        assert attrs["slm.max_tokens"] == 100
        assert attrs["slm.temperature"] == 0.7
        assert attrs["slm.input.messages"] == 2
        assert attrs["slm.input.content_length"] == 13  # "Hello" + "Hi there"
    
    def test_estimates_prompt_tokens_for_streaming(self, setup_tracing):
        """Test that prompt tokens are estimated for streaming."""
        memory_exporter, local_tracer = setup_tracing
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="A" * 40)],  # 40 chars
            stream=True
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_span(request, is_streaming=True) as (span, messages):
                pass
        
        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        
        # Should estimate ~10 tokens (40 chars / 4)
        assert span.attributes["slm.usage.prompt_tokens"] == 10
    
    def test_handles_exceptions(self, setup_tracing):
        """Test exception handling in span context."""
        memory_exporter, local_tracer = setup_tracing
        
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="test")])
        
        with pytest.raises(ValueError):
            # Patch the global tracer with our local one
            with patch('slm_server.utils.spans.tracer', local_tracer):
                with slm_span(request, is_streaming=False) as (span, messages):
                    raise ValueError("test error")
        
        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        
        assert span.status.status_code == StatusCode.ERROR
        assert "test error" in span.status.description
        assert span.attributes["slm.force_sample"] is True


class TestSLMLoggingSpanProcessor:
    """Tests for SLMLoggingSpanProcessor."""
    
    def test_logs_span_start(self):
        """Test logging of span start."""
        with patch('slm_server.utils.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            processor = SLMLoggingSpanProcessor()
            
            mock_span = Mock()
            mock_span.name = "slm.chat_completion.streaming"
            mock_span.attributes = {
                "slm.max_tokens": 100,
                "slm.temperature": 0.7,
                "slm.input.messages": 2,
                "slm.input.content_length": 15,
                "slm.streaming": True,
            }
            
            processor.on_start(mock_span)
            
            # Verify logging call
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "streaming" in call_args
    
    def test_logs_span_completion(self):
        """Test logging of successful span completion."""
        with patch('slm_server.utils.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            processor = SLMLoggingSpanProcessor()
            
            mock_span = Mock()
            mock_span.name = "slm.chat_completion.non_streaming"
            mock_span.status = Mock()
            mock_span.status.status_code = StatusCode.OK
            mock_span.start_time = 1000_000_000
            mock_span.end_time = 2000_000_000
            mock_span.attributes = {
                "slm.streaming": False,
                "slm.usage.total_tokens": 50,
                "slm.usage.completion_tokens": 25,
                "slm.output.content_length": 100,
            }
            mock_span.events = []
            
            processor.on_end(mock_span)
            
            # Verify logging call
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "completed" in call_args.lower()
    
    def test_logs_span_error(self):
        """Test logging of span errors."""
        with patch('slm_server.utils.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            processor = SLMLoggingSpanProcessor()
            
            mock_span = Mock()
            mock_span.name = "slm.chat_completion.streaming"
            mock_span.status = Mock()
            mock_span.status.status_code = StatusCode.ERROR
            mock_span.status.description = "Test error"
            
            processor.on_end(mock_span)
            
            # Verify error logging
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "failed" in call_args.lower()


class TestSLMMetricsSpanProcessor:
    """Tests for SLMMetricsSpanProcessor."""
    
    def test_records_completion_metrics(self):
        """Test recording of completion duration metrics."""
        from prometheus_client import REGISTRY
        # Clear any existing metrics to avoid conflicts
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
        
        processor = SLMMetricsSpanProcessor()
        
        mock_span = Mock()
        mock_span.name = "slm.chat_completion.non_streaming"
        mock_span.status = Mock()
        mock_span.status.status_code = StatusCode.OK
        mock_span.start_time = 1000_000_000
        mock_span.end_time = 2000_000_000
        mock_span.attributes = {
            "slm.model": "test-model",
            "slm.streaming": False,
            "slm.usage.prompt_tokens": 10,
            "slm.usage.completion_tokens": 20,
            "slm.usage.total_tokens": 30,
        }
        mock_span.events = []
        
        # Mock the histogram observe method
        with patch.object(processor.completion_duration, 'labels') as mock_labels:
            mock_metric = Mock()
            mock_labels.return_value = mock_metric
            
            processor.on_end(mock_span)
            
            # Verify metric was recorded
            mock_labels.assert_called_with(
                model="test-model", 
                streaming="non_streaming", 
                status="success"
            )
            mock_metric.observe.assert_called_with(1.0)  # 1 second
    
    def test_records_token_metrics(self):
        """Test recording of token count metrics."""
        from prometheus_client import REGISTRY
        # Clear any existing metrics to avoid conflicts
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
        
        processor = SLMMetricsSpanProcessor()
        
        mock_span = Mock()
        mock_span.name = "slm.chat_completion.streaming"
        mock_span.status = Mock()
        mock_span.status.status_code = StatusCode.OK
        mock_span.start_time = 1000_000_000
        mock_span.end_time = 2000_000_000
        mock_span.attributes = {
            "slm.model": "test-model",
            "slm.streaming": True,
            "slm.usage.prompt_tokens": 15,
            "slm.usage.completion_tokens": 25,
        }
        mock_span.events = []
        
        # Mock the histogram observe methods
        with patch.object(processor.token_count, 'labels') as mock_token_labels:
            mock_token_metric = Mock()
            mock_token_labels.return_value = mock_token_metric
            
            processor.on_end(mock_span)
            
            # Verify both prompt and completion token metrics were recorded
            mock_token_labels.assert_any_call(
                model="test-model", 
                streaming="streaming", 
                token_type="prompt"
            )
            mock_token_labels.assert_any_call(
                model="test-model", 
                streaming="streaming", 
                token_type="completion"
            )
    
    def test_records_error_metrics(self):
        """Test recording of error metrics."""
        from prometheus_client import REGISTRY
        # Clear any existing metrics to avoid conflicts
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
        
        processor = SLMMetricsSpanProcessor()
        
        mock_span = Mock()
        mock_span.name = "slm.chat_completion.non_streaming"
        mock_span.status = Mock()
        mock_span.status.status_code = StatusCode.ERROR
        mock_span.status.description = "Test error"
        mock_span.start_time = None  # Set to None to trigger early return
        mock_span.end_time = None
        mock_span.attributes = {
            "slm.model": "test-model",
            "slm.streaming": False,
        }
        
        # Mock the counter inc method
        with patch.object(processor.error_total, 'labels') as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter
            
            processor.on_end(mock_span)
            
            # Verify error metric was recorded
            mock_labels.assert_called_with(
                model="test-model", 
                streaming="non_streaming", 
            )
            mock_counter.inc.assert_called_once()


class TestIntegrationStreamingCall:
    """Integration test for streaming call flow."""
    
    def test_complete_streaming_flow(self, setup_tracing):
        """Test complete flow of streaming call with events and metrics."""
        memory_exporter, local_tracer = setup_tracing
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello world")],
            stream=True,
            max_tokens=50,
            temperature=0.8
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_span(request, is_streaming=True) as (span, messages_for_llm):
                # Simulate processing chunks
                chunks = [
                    ChatCompletionStreamResponse(
                        model="test-model",
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(content="Hi"),
                            finish_reason=None
                        )]
                    ),
                    ChatCompletionStreamResponse(
                        model="test-model", 
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=" there!"),
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
                
                for chunk in chunks:
                    set_atrribute_response_stream(span, chunk)
        
        # Get finished span and verify
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        
        # Verify span attributes
        assert finished_span.attributes["slm.streaming"] is True
        assert finished_span.attributes["slm.usage.completion_tokens"] == 2  # "Hi" + " there!"
        assert finished_span.attributes["slm.output.content_length"] == 9  # "Hi there!"
        assert finished_span.attributes["slm.output.chunk_count"] == 2  # Only chunks with content
        
        # Verify events were recorded
        chunk_events = [e for e in finished_span.events if e.name == EVENT_CHUNK_GENERATED]
        assert len(chunk_events) == 3  # All chunks including empty
        
        # Verify performance metrics can be calculated
        metrics = calculate_performance_metrics(finished_span)
        assert METRIC_TOTAL_DURATION in metrics
        assert METRIC_CHUNKS_WITH_CONTENT in metrics
        assert metrics[METRIC_CHUNKS_WITH_CONTENT] == 2
        assert metrics[METRIC_EMPTY_CHUNKS] == 1


class TestIntegrationNonStreamingCall:
    """Integration test for non-streaming call flow."""
    
    def test_complete_non_streaming_flow(self, setup_tracing):
        """Test complete flow of non-streaming call."""
        memory_exporter, local_tracer = setup_tracing
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test message")],
            stream=False,
            max_tokens=100,
            temperature=0.5
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_span(request, is_streaming=False) as (span, messages_for_llm):
                # Simulate processing response
                response = ChatCompletionResponse(
                    model="test-model",
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="Test response content"),
                        finish_reason="stop"
                    )],
                    usage=Usage(prompt_tokens=12, completion_tokens=18, total_tokens=30)
                )
                
                set_atrribute_response(span, response)
        
        # Get finished span and verify
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        
        # Verify span attributes
        assert finished_span.attributes["slm.streaming"] is False
        assert finished_span.attributes["slm.usage.prompt_tokens"] == 12
        assert finished_span.attributes["slm.usage.completion_tokens"] == 18
        assert finished_span.attributes["slm.usage.total_tokens"] == 30
        assert finished_span.attributes["slm.output.content_length"] == 21  # "Test response content"
        
        # Verify no chunk events for non-streaming
        chunk_events = [e for e in finished_span.events if e.name == EVENT_CHUNK_GENERATED]
        assert len(chunk_events) == 0
        
        # Verify performance metrics
        metrics = calculate_performance_metrics(finished_span)
        assert METRIC_TOTAL_DURATION in metrics
        assert METRIC_TOKENS_PER_SECOND in metrics
        assert METRIC_TOTAL_TOKENS_PER_SECOND in metrics