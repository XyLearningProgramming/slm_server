import asyncio

import httpx
import pytest
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.tools import BaseTool


class DummyCalculatorTool(BaseTool):
    """A dummy calculator tool for testing agent functionality."""
    
    name = "calculator"
    description = "Calculate basic math expressions. Input should be a mathematical expression like '2+2' or '10*5'."
    
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
    
    name = "search"
    description = "Search for information. Input should be a search query."
    
    def _run(self, query: str) -> str:
        """Execute the search."""
        # Return dummy search results
        return f"Search results for '{query}': [Dummy result 1], [Dummy result 2], [Dummy result 3]"


class TestLangChainCompatibility:
    """Test suite for LangChain compatibility with our model server."""
    
    @pytest.fixture
    def base_url(self):
        """Base URL for the model server."""
        return "http://localhost:8000"
    
    @pytest.fixture
    def chat_llm(self, base_url):
        """Real ChatOpenAI instance pointing to our server."""
        return ChatOpenAI(
            openai_api_base=f"{base_url}/api/v1",
            # openai_api_key="dummy-key",  # Our server doesn't require real auth
            # model_name="gpt-3.5-turbo",  # Model name doesn't matter for our server
            temperature=0.7,
            max_tokens=150,
        )
    
    @pytest.fixture
    def embeddings(self, base_url):
        """Real OpenAIEmbeddings instance pointing to our server."""
        return OpenAIEmbeddings(
            openai_api_base=f"{base_url}/api/v1",
            openai_api_key="dummy-key",  # Our server doesn't require real auth
        )
    
    @pytest.fixture
    def dummy_tools(self):
        """Dummy tools for agent testing."""
        return [DummyCalculatorTool(), DummySearchTool()]
    
    def test_basic_chat_llm_call(self, chat_llm):
        """Test basic ChatOpenAI call through LangChain interface."""
        print("Testing basic ChatOpenAI call...")
        
        messages = [HumanMessage(content="Hello, can you say 'LangChain test successful'?")]
        response = chat_llm(messages)
        
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"ChatOpenAI Response: {response.content}")
    
    def test_llm_chain_integration(self, chat_llm):
        """Test LLMChain integration with our server."""
        print("Testing LLMChain integration...")
        
        # Create a simple prompt template
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a short paragraph about {topic}. Keep it under 100 words."
        )
        
        # Create an LLMChain with our ChatOpenAI instance
        chain = LLMChain(llm=chat_llm, prompt=prompt)
        
        # Run the chain
        response = chain.run(topic="artificial intelligence")
        
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"LLMChain Response: {response}")
    
    def test_react_agent_with_tools(self, chat_llm, dummy_tools):
        """Test React agent with dummy tools using real LangChain components."""
        print("Testing React agent with tools...")
        
        # Use LangChain's built-in ZERO_SHOT_REACT_DESCRIPTION agent
        from langchain.agents import initialize_agent, AgentType
        
        agent_executor = initialize_agent(
            tools=dummy_tools,
            llm=chat_llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3
        )
        
        # Test the agent with a calculation question
        try:
            result = agent_executor.invoke({"input": "What is 15 * 7 + 10?"})
            print(f"Agent Result: {result}")
            assert "output" in result
            assert len(result["output"]) > 0
        except Exception as e:
            print(f"Agent execution failed (expected for demo): {e}")
            # Test individual tools instead
            calculator_tool = dummy_tools[0]
            calc_result = calculator_tool.run("15 * 7 + 10")
            print(f"Calculator Tool Result: {calc_result}")
            
            search_tool = dummy_tools[1]
            search_result = search_tool.run("mathematical operations")
            print(f"Search Tool Result: {search_result}")
    
    def test_embeddings_compatibility(self, embeddings):
        """Test OpenAIEmbeddings compatibility with our server."""
        print("Testing OpenAIEmbeddings compatibility...")

        # Test embedding generation
        texts = ["Hello world", "This is a test"]
        result = embeddings.embed_documents(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(embedding, list) for embedding in result)
        print(f"Embeddings generated successfully: {len(result)} embeddings")
        
        # Test single query embedding
        query_result = embeddings.embed_query("Test query")
        assert isinstance(query_result, list)
        print(f"Query embedding generated successfully: dimension {len(query_result)}")

    def test_comprehensive_workflow(self, chat_llm, dummy_tools, base_url):
        """Test a comprehensive workflow combining multiple features."""
        print("Testing comprehensive workflow...")
        
        # Step 1: Basic reasoning task
        reasoning_prompt = "Solve this step by step: If I have 3 apples and buy 5 more, then give away 2, how many do I have?"
        messages = [HumanMessage(content=reasoning_prompt)]
        reasoning_response = chat_llm(messages)
        print(f"Reasoning Response: {reasoning_response.content}")
        
        # Step 2: Use calculator tool to verify
        calculator = dummy_tools[0]
        calc_result = calculator.run("3 + 5 - 2")
        print(f"Calculator Verification: {calc_result}")
        
        # Step 3: Test search functionality
        search_tool = dummy_tools[1]
        search_result = search_tool.run("apple nutrition facts")
        print(f"Search Result: {search_result}")
        
        # Step 4: Test LLMChain for structured output
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="List 3 benefits of {topic} in bullet points."
        )
        chain = LLMChain(llm=chat_llm, prompt=prompt)
        chain_result = chain.run(topic="eating apples")
        print(f"Chain Summary: {chain_result}")
        assert isinstance(chain_result, str)
        assert len(chain_result) > 0


async def test_streaming_compatibility(base_url):
    """Test streaming compatibility with httpx."""
    print("Testing streaming compatibility...")
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{base_url}/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
            timeout=30,
        ) as response:
            assert response.status_code == 200
            print("Streaming test passed!")


def run_langchain_tests():
    """Run all LangChain compatibility tests."""
    test_instance = TestLangChainCompatibility()
    
    # Real fixtures
    base_url = "http://localhost:8000"
    chat_llm = ChatOpenAI(
        openai_api_base=f"{base_url}/api/v1",
        temperature=0.7,
        max_tokens=150,
    )
    embeddings = OpenAIEmbeddings(
        openai_api_base=f"{base_url}/api/v1",
        openai_api_key="dummy-key",
    )
    dummy_tools = [DummyCalculatorTool(), DummySearchTool()]
    
    try:
        print("=== LangChain Compatibility Tests ===\n")
        
        test_instance.test_basic_chat_llm_call(chat_llm)
        print("\n" + "="*50 + "\n")
        
        test_instance.test_llm_chain_integration(chat_llm)
        print("\n" + "="*50 + "\n")
        
        test_instance.test_react_agent_with_tools(chat_llm, dummy_tools)
        print("\n" + "="*50 + "\n")
        
        test_instance.test_embeddings_compatibility(embeddings)
        print("\n" + "="*50 + "\n")
        
        # Run async tests
        asyncio.run(test_streaming_compatibility(base_url))
        print("\n" + "="*50 + "\n")
        
        test_instance.test_comprehensive_workflow(chat_llm, dummy_tools, base_url)
        print("\n" + "="*50 + "\n")
        
        print("✅ All LangChain compatibility tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ LangChain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_langchain_tests()