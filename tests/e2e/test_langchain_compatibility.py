
import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool


class DummyCalculatorTool(BaseTool):
    """A dummy calculator tool for testing agent functionality."""

    name: str = "calculator"
    description: str = "Calculate basic math expressions. Input should be a mathematical expression like '2+2' or '10*5'."

    def _run(self, query: str) -> str:
        """Execute the calculation."""
        try:
            # Simple eval for basic math (in production, use a proper math parser)
            result = eval(query.strip())
            return f"The result is: {result}"
        except Exception as e:
            return f"Error calculating {query}: {str(e)}"


class DummySearchTool(BaseTool):
    """A dummy search tool for testing agent functionality."""

    name: str = "search"
    description: str = "Search for information. Input should be a search query."

    def _run(self, query: str) -> str:
        """Execute the search."""
        # Return dummy search results
        return f"Search results for '{query}': [Dummy result 1], [Dummy result 2], [Dummy result 3]"

@pytest.mark.langchain
def test_basic_chat_llm_call(server):
    """Test basic ChatOpenAI call through LangChain interface."""
    chat_llm = ChatOpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="dummy-key",
        temperature=0.7,
        max_tokens=150,
    )
    messages = [HumanMessage(content="Hello, can you say 'LangChain test successful'?")]
    response = chat_llm.invoke(messages)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    print(f"TEST LANGCHAIN RESPONSE: {response.content}")

@pytest.mark.langchain
def test_llm_chain_integration(server):
    """Test modern RunnableSequence chain integration with our server."""
    chat_llm = ChatOpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="dummy-key",
        temperature=0.7,
        max_tokens=150,
    )
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short paragraph about {topic}. Keep it under 100 words."
    )
    chain = prompt | chat_llm
    response = chain.invoke({"topic": "artificial intelligence"})
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    print(f"TEST LANGCHAIN RESPONSE: {response.content}")


@pytest.mark.langchain
def test_embeddings_compatibility(server):
    """Test OpenAIEmbeddings compatibility with our server."""
    embeddings = OpenAIEmbeddings(
        base_url="http://localhost:8000/api/v1",
        api_key="dummy-key",
    )
    texts = ["Hello world", "This is a test"]
    result = embeddings.embed_documents(texts)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(embedding, list) for embedding in result)

    query_result = embeddings.embed_query("Test query")
    assert isinstance(query_result, list)


@pytest.mark.langchain
def test_react_agent_with_tools(server):
    """Test ReAct agent with tools using modern LangGraph."""
    from langgraph.prebuilt import create_react_agent
    chat_llm = ChatOpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="dummy-key",
        temperature=0.7,
        max_tokens=150,
    )
    tools = [DummyCalculatorTool(), DummySearchTool()]
    
    # Use LangGraph's prebuilt ReAct agent
    agent_executor = create_react_agent(chat_llm, tools)
    
    # try:
    # LangGraph agents use a different input format
    result = agent_executor.invoke({"input": "Can you search for what AI is using tool and trust its results?"})
    # Check that we got a response
    assert "messages" in result
    assert len(result["messages"]) > 0
    # The last message should be the agent's final response
    final_message = result["messages"][-1]
    assert hasattr(final_message, 'content')
    assert len(final_message.content) > 0
    print(f"TEST LANGCHAIN RESPONSE: {result}")
    # except Exception as e:
    #     # The agent may fail with a simple model, which is expected.
    #     # We still want to ensure the tools themselves work.
    #     print(f"Agent execution failed as expected: {e}")
    #     calculator_tool = tools[0]
    #     calc_result = calculator_tool.run("15 * 7 + 10")
    #     assert "115" in calc_result

    #     search_tool = tools[1]
    #     search_result = search_tool.run("langchain")
    #     assert "Dummy result" in search_result
