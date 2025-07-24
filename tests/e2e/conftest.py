
import socket
import subprocess
import time
import pytest
import httpx

def is_server_running(port=8000):
    """Check if a server is running on the specified port by checking the /health endpoint."""
    try:
        response = httpx.get(f"http://localhost:{port}/health", timeout=1)
        return response.status_code == 200
    except httpx.RequestError:
        return False

@pytest.fixture(scope="session")
def server():
    """
    A session-scoped fixture that starts the SLM server if it's not already running.
    It tears down the server process after all tests in the session are complete.
    """
    if is_server_running():
        print("Server is already running. Tests will proceed against the existing server.")
        yield
        return

    print("Starting server...")
    # Start the server as a background process
    process = subprocess.Popen(["./scripts/start.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for the server to be ready
    for _ in range(30):  # 30 seconds timeout
        if is_server_running():
            print("Server started successfully.")
            break
        time.sleep(1)
    else:
        stdout, stderr = process.communicate()
        print(f"Server failed to start. Stdout: {stdout.decode()}, Stderr: {stderr.decode()}")
        pytest.fail("Server did not start within the timeout period.", pytrace=False)

    yield

    print("Tearing down server...")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("Server did not terminate gracefully, killing it.")
        process.kill()
    print("Server torn down.")
