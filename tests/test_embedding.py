"""Tests for embedding functionality in slm_server."""

from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from slm_server.model import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from slm_server.utils import (
    ATTR_INPUT_COUNT,
    ATTR_INPUT_CONTENT_LENGTH,
    ATTR_MODEL,
    ATTR_OUTPUT_COUNT,
    ATTR_PROMPT_TOKENS,
    ATTR_TOTAL_TOKENS,
    SPAN_EMBEDDING,
    set_attribute_response_embedding,
    slm_embedding_span,
)


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


class TestSetAttributeResponseEmbedding:
    """Tests for set_attribute_response_embedding function."""
    
    def test_sets_embedding_attributes_correctly(self):
        """Test that embedding response attributes are set correctly on span."""
        mock_span = Mock()
        
        # Create embedding response with usage and data
        response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=[0.1, 0.2, -0.3, 0.4, -0.5],
                    index=0
                ),
                EmbeddingData(
                    object="embedding", 
                    embedding=[0.6, -0.7, 0.8, -0.9, 1.0],
                    index=1
                )
            ],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=15, total_tokens=15)
        )
        
        set_attribute_response_embedding(mock_span, response)
        
        # Verify attributes were set
        mock_span.set_attribute.assert_any_call(ATTR_PROMPT_TOKENS, 15)
        mock_span.set_attribute.assert_any_call(ATTR_TOTAL_TOKENS, 15)
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_COUNT, 2)  # 2 embeddings
    
    def test_handles_single_embedding(self):
        """Test handling of single embedding response."""
        mock_span = Mock()
        
        response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=[0.1, 0.2, 0.3],
                    index=0
                )
            ],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5)
        )
        
        set_attribute_response_embedding(mock_span, response)
        
        # Should set output count to 1
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_COUNT, 1)
    
    def test_handles_empty_data(self):
        """Test handling of empty embedding data."""
        mock_span = Mock()
        
        response = EmbeddingResponse(
            object="list",
            data=[],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0)
        )
        
        set_attribute_response_embedding(mock_span, response)
        
        # Should still set usage attributes but not output count since data is empty
        mock_span.set_attribute.assert_any_call(ATTR_PROMPT_TOKENS, 0)
        mock_span.set_attribute.assert_any_call(ATTR_TOTAL_TOKENS, 0)
        # Verify output count was NOT set since data is empty
        output_count_calls = [call for call in mock_span.set_attribute.call_args_list 
                             if call[0][0] == ATTR_OUTPUT_COUNT]
        assert len(output_count_calls) == 0
    
    def test_handles_usage_properly(self):
        """Test that usage attributes are set when present."""
        mock_span = Mock()
        
        response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=[0.1, 0.2],
                    index=0
                )
            ],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5)
        )
        
        set_attribute_response_embedding(mock_span, response)
        
        # Should set both usage and output count attributes
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_COUNT, 1)
        mock_span.set_attribute.assert_any_call(ATTR_PROMPT_TOKENS, 5)
        mock_span.set_attribute.assert_any_call(ATTR_TOTAL_TOKENS, 5)


class TestSlmEmbeddingSpan:
    """Tests for slm_embedding_span context manager."""
    
    def test_sets_initial_attributes_string_input(self, setup_tracing):
        """Test that initial attributes are set correctly for string input."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input="Hello world, this is a test input.",
            model="test-model"
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request) as span:
                pass
        
        # Get the finished span
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        attrs = span.attributes
        
        assert span.name == SPAN_EMBEDDING
        assert attrs[ATTR_MODEL] == "llama-cpp"
        assert attrs[ATTR_INPUT_COUNT] == 1
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] > 0
    
    def test_sets_initial_attributes_list_input(self, setup_tracing):
        """Test that initial attributes are set correctly for list input."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input=["First text", "Second text", "Third text"],
            model="test-model"
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request) as span:
                pass
        
        # Get the finished span
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        attrs = span.attributes
        
        assert attrs[ATTR_INPUT_COUNT] == 3
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] > 0
    
    def test_handles_empty_string_input(self, setup_tracing):
        """Test handling of empty string input."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input="",
            model="test-model"
        )
        
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request) as span:
                pass
        
        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        assert attrs[ATTR_INPUT_COUNT] == 1
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] == 0
    
    def test_handles_empty_list_input(self, setup_tracing):
        """Test handling of empty list input."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input=[],
            model="test-model"
        )
        
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request):
                pass
        
        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        assert attrs[ATTR_INPUT_COUNT] == 0
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] == 0
    
    def test_handles_list_with_empty_strings(self, setup_tracing):
        """Test handling of list containing empty strings."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input=["Hello", "", "World", ""],
            model="test-model"
        )
        
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request) as span:
                pass
        
        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        attrs = span.attributes
        
        assert attrs[ATTR_INPUT_COUNT] == 4
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] == 10  # len("Hello") + len("World") = 5 + 5
    
    def test_handles_exceptions(self, setup_tracing):
        """Test exception handling in embedding span context."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(input="test", model="test-model")
        
        with pytest.raises(ValueError):
            with patch('slm_server.utils.spans.tracer', local_tracer):
                with slm_embedding_span(request) as span:
                    raise ValueError("test embedding error")
        
        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        
        assert span.status.status_code == StatusCode.ERROR
        assert "test embedding error" in span.status.description
        assert span.attributes["slm.force_sample"] is True


class TestEmbeddingModelValidation:
    """Tests for embedding model validation."""
    
    def test_embedding_request_string_input(self):
        """Test EmbeddingRequest with string input."""
        request = EmbeddingRequest(
            input="Test input text",
            model="test-model"
        )
        
        assert request.input == "Test input text"
        assert request.model == "test-model"
    
    def test_embedding_request_list_input(self):
        """Test EmbeddingRequest with list input."""
        request = EmbeddingRequest(
            input=["First", "Second", "Third"],
            model="test-model"
        )
        
        assert request.input == ["First", "Second", "Third"]
        assert request.model == "test-model"
    
    def test_embedding_request_default_model(self):
        """Test EmbeddingRequest with default model."""
        request = EmbeddingRequest(input="Test")
        
        assert request.model == "Qwen3-0.6B-GGUF"  # Default from model definition
    
    def test_embedding_response_creation(self):
        """Test EmbeddingResponse creation."""
        response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=[1.0, 2.0, 3.0],
                    index=0
                )
            ],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10)
        )
        
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].embedding == [1.0, 2.0, 3.0]
        assert response.data[0].index == 0
        assert response.model == "test-model"
        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 10
    
    def test_embedding_data_defaults(self):
        """Test EmbeddingData with default values."""
        data = EmbeddingData(
            embedding=[0.1, 0.2, 0.3],
            index=0
        )
        
        assert data.object == "embedding"  # Default value
        assert data.embedding == [0.1, 0.2, 0.3]
        assert data.index == 0


class TestIntegrationEmbeddingFlow:
    """Integration test for complete embedding flow."""
    
    def test_complete_embedding_flow_string_input(self, setup_tracing):
        """Test complete flow of embedding request with string input."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input="This is a test sentence for embedding.",
            model="test-model"
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request) as span:
                # Simulate processing embedding
                response = EmbeddingResponse(
                    object="list",
                    data=[
                        EmbeddingData(
                            object="embedding",
                            embedding=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
                            index=0
                        )
                    ],
                    model="test-model",
                    usage=EmbeddingUsage(prompt_tokens=8, total_tokens=8)
                )
                
                set_attribute_response_embedding(span, response)
        
        # Get finished span and verify
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        
        # Verify span attributes
        assert finished_span.name == SPAN_EMBEDDING
        assert finished_span.attributes[ATTR_MODEL] == "llama-cpp"
        assert finished_span.attributes[ATTR_INPUT_COUNT] == 1
        assert finished_span.attributes[ATTR_INPUT_CONTENT_LENGTH] > 0
        assert finished_span.attributes[ATTR_OUTPUT_COUNT] == 1
        assert finished_span.attributes[ATTR_PROMPT_TOKENS] == 8
        assert finished_span.attributes[ATTR_TOTAL_TOKENS] == 8
    
    def test_complete_embedding_flow_list_input(self, setup_tracing):
        """Test complete flow of embedding request with list input."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input=["First sentence.", "Second sentence.", "Third sentence."],
            model="test-model"
        )
        
        # Patch the global tracer with our local one
        with patch('slm_server.utils.spans.tracer', local_tracer):
            with slm_embedding_span(request) as span:
                # Simulate processing multiple embeddings
                response = EmbeddingResponse(
                    object="list",
                    data=[
                        EmbeddingData(
                            object="embedding",
                            embedding=[0.1, 0.2, 0.3],
                            index=0
                        ),
                        EmbeddingData(
                            object="embedding",
                            embedding=[0.4, 0.5, 0.6],
                            index=1
                        ),
                        EmbeddingData(
                            object="embedding",
                            embedding=[0.7, 0.8, 0.9],
                            index=2
                        )
                    ],
                    model="test-model",
                    usage=EmbeddingUsage(prompt_tokens=12, total_tokens=12)
                )
                
                set_attribute_response_embedding(span, response)
        
        # Get finished span and verify
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        
        # Verify span attributes
        assert finished_span.attributes[ATTR_INPUT_COUNT] == 3
        assert finished_span.attributes[ATTR_INPUT_CONTENT_LENGTH] > 0
        assert finished_span.attributes[ATTR_OUTPUT_COUNT] == 3
        assert finished_span.attributes[ATTR_PROMPT_TOKENS] == 12
        assert finished_span.attributes[ATTR_TOTAL_TOKENS] == 12
    
    def test_embedding_flow_with_error(self, setup_tracing):
        """Test embedding flow with error handling."""
        memory_exporter, local_tracer = setup_tracing
        
        request = EmbeddingRequest(
            input="This will cause an error.",
            model="test-model"
        )
        
        with pytest.raises(RuntimeError):
            with patch('slm_server.utils.spans.tracer', local_tracer):
                with slm_embedding_span(request) as span:
                    raise RuntimeError("Embedding processing failed")
        
        # Get finished span and verify error handling
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        
        # Verify error status
        assert finished_span.status.status_code == StatusCode.ERROR
        assert "Embedding processing failed" in finished_span.status.description
        assert finished_span.attributes["slm.force_sample"] is True
        
        # Initial attributes should still be set
        assert finished_span.attributes[ATTR_INPUT_COUNT] == 1
        assert finished_span.attributes[ATTR_INPUT_CONTENT_LENGTH] == 25