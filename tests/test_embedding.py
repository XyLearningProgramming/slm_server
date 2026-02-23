"""Tests for embedding functionality in slm_server."""

from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from slm_server.model import EmbeddingData, EmbeddingRequest, EmbeddingResponse
from slm_server.utils import (
    ATTR_INPUT_COUNT,
    ATTR_INPUT_CONTENT_LENGTH,
    ATTR_MODEL,
    ATTR_OUTPUT_COUNT,
    EMBEDDING_MODEL_NAME,
    SPAN_EMBEDDING,
    set_attribute_response_embedding,
    slm_embedding_span,
)


@pytest.fixture
def setup_tracing():
    """Set up tracing with in-memory span exporter for testing."""
    tracer_provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    span_processor = SimpleSpanProcessor(memory_exporter)
    tracer_provider.add_span_processor(span_processor)

    local_tracer = tracer_provider.get_tracer(__name__)

    yield memory_exporter, local_tracer

    memory_exporter.clear()


class TestSetAttributeResponseEmbedding:
    """Tests for set_attribute_response_embedding function."""

    def test_sets_output_count(self):
        mock_span = Mock()
        response = EmbeddingResponse(
            data=[
                EmbeddingData(embedding=[0.1, 0.2], index=0),
                EmbeddingData(embedding=[0.3, 0.4], index=1),
            ],
            model="test-model",
        )
        set_attribute_response_embedding(mock_span, response)
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_COUNT, 2)

    def test_handles_single_embedding(self):
        mock_span = Mock()
        response = EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)],
            model="test-model",
        )
        set_attribute_response_embedding(mock_span, response)
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_COUNT, 1)

    def test_handles_empty_data(self):
        mock_span = Mock()
        response = EmbeddingResponse(data=[], model="test-model")
        set_attribute_response_embedding(mock_span, response)
        output_count_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == ATTR_OUTPUT_COUNT
        ]
        assert len(output_count_calls) == 0


class TestSlmEmbeddingSpan:
    """Tests for slm_embedding_span context manager."""

    def test_sets_initial_attributes_string_input(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(
            input="Hello world, this is a test input.", model="test-model"
        )

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request) as span:
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attrs = span.attributes

        assert span.name == SPAN_EMBEDDING
        assert attrs[ATTR_MODEL] == EMBEDDING_MODEL_NAME
        assert attrs[ATTR_INPUT_COUNT] == 1
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] > 0

    def test_sets_initial_attributes_list_input(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(
            input=["First text", "Second text", "Third text"], model="test-model"
        )

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request) as span:
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attrs = span.attributes

        assert attrs[ATTR_INPUT_COUNT] == 3
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] > 0

    def test_handles_empty_string_input(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(input="", model="test-model")

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request) as span:
                pass

        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        attrs = span.attributes

        assert attrs[ATTR_INPUT_COUNT] == 1
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] == 0

    def test_handles_empty_list_input(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(input=[], model="test-model")

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request):
                pass

        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        attrs = span.attributes

        assert attrs[ATTR_INPUT_COUNT] == 0
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] == 0

    def test_handles_list_with_empty_strings(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(
            input=["Hello", "", "World", ""], model="test-model"
        )

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request) as span:
                pass

        spans = memory_exporter.get_finished_spans()
        span = spans[0]
        attrs = span.attributes

        assert attrs[ATTR_INPUT_COUNT] == 4
        assert attrs[ATTR_INPUT_CONTENT_LENGTH] == 10

    def test_handles_exceptions(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(input="test", model="test-model")

        with pytest.raises(ValueError):
            with patch("slm_server.utils.spans.tracer", local_tracer):
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
        request = EmbeddingRequest(input="Test input text", model="test-model")
        assert request.input == "Test input text"
        assert request.model == "test-model"

    def test_embedding_request_list_input(self):
        request = EmbeddingRequest(
            input=["First", "Second", "Third"], model="test-model"
        )
        assert request.input == ["First", "Second", "Third"]
        assert request.model == "test-model"

    def test_embedding_request_default_model(self):
        request = EmbeddingRequest(input="Test")
        assert request.model is None

    def test_embedding_response_creation(self):
        response = EmbeddingResponse(
            data=[EmbeddingData(embedding=[1.0, 2.0, 3.0], index=0)],
            model="test-model",
        )
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].embedding == [1.0, 2.0, 3.0]
        assert response.data[0].index == 0
        assert response.model == "test-model"

    def test_embedding_data_defaults(self):
        data = EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)
        assert data.object == "embedding"
        assert data.embedding == [0.1, 0.2, 0.3]
        assert data.index == 0


class TestIntegrationEmbeddingFlow:
    """Integration test for complete embedding flow."""

    def test_complete_embedding_flow_string_input(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(
            input="This is a test sentence for embedding.", model="test-model"
        )

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request) as span:
                response = EmbeddingResponse(
                    data=[
                        EmbeddingData(
                            embedding=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
                            index=0,
                        )
                    ],
                    model="test-model",
                )
                set_attribute_response_embedding(span, response)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        finished_span = spans[0]

        assert finished_span.name == SPAN_EMBEDDING
        assert finished_span.attributes[ATTR_MODEL] == EMBEDDING_MODEL_NAME
        assert finished_span.attributes[ATTR_INPUT_COUNT] == 1
        assert finished_span.attributes[ATTR_INPUT_CONTENT_LENGTH] > 0
        assert finished_span.attributes[ATTR_OUTPUT_COUNT] == 1

    def test_complete_embedding_flow_list_input(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(
            input=["First sentence.", "Second sentence.", "Third sentence."],
            model="test-model",
        )

        with patch("slm_server.utils.spans.tracer", local_tracer):
            with slm_embedding_span(request) as span:
                response = EmbeddingResponse(
                    data=[
                        EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
                        EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
                        EmbeddingData(embedding=[0.7, 0.8, 0.9], index=2),
                    ],
                    model="test-model",
                )
                set_attribute_response_embedding(span, response)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        finished_span = spans[0]

        assert finished_span.attributes[ATTR_INPUT_COUNT] == 3
        assert finished_span.attributes[ATTR_INPUT_CONTENT_LENGTH] > 0
        assert finished_span.attributes[ATTR_OUTPUT_COUNT] == 3

    def test_embedding_flow_with_error(self, setup_tracing):
        memory_exporter, local_tracer = setup_tracing

        request = EmbeddingRequest(
            input="This will cause an error.", model="test-model"
        )

        with pytest.raises(RuntimeError):
            with patch("slm_server.utils.spans.tracer", local_tracer):
                with slm_embedding_span(request) as span:
                    raise RuntimeError("Embedding processing failed")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        finished_span = spans[0]

        assert finished_span.status.status_code == StatusCode.ERROR
        assert "Embedding processing failed" in finished_span.status.description
        assert finished_span.attributes["slm.force_sample"] is True
        assert finished_span.attributes[ATTR_INPUT_COUNT] == 1
        assert finished_span.attributes[ATTR_INPUT_CONTENT_LENGTH] == 25
