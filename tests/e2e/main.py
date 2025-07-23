import argparse
import asyncio

from test_api import run_api_tests
from test_langchain_compatibility import run_langchain_tests


async def main():
    """Main entry point with argument parsing for test groups."""
    parser = argparse.ArgumentParser(description="Run e2e tests")
    parser.add_argument(
        "--skip", 
        action="append",
        choices=["api", "langchain"],
        help="Skip specific test groups (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    skip_groups = args.skip or []
    run_api = "api" not in skip_groups
    run_langchain = "langchain" not in skip_groups
    
    success = True
    
    if run_api:
        print("Starting API tests...")
        api_success = await run_api_tests()
        success = success and api_success
        print()
    
    if run_langchain:
        print("Starting LangChain compatibility tests...")
        langchain_success = run_langchain_tests()
        success = success and langchain_success
        print()

    
    if success:
        print("🎉 All selected tests completed successfully!")
    else:
        raise Exception("❌ Some tests failed!")


if __name__ == "__main__":
    asyncio.run(main())