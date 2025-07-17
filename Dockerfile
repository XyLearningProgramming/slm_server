# 1. Use a slim Python 3.13 image, matching the project's requirement
FROM python:3.13-slim

# 2. Set Environment Variables
#    - WORKER_COUNT: Number of uvicorn workers, defaults to 1.
#      Can be overridden at runtime (e.g., docker run -e WORKER_COUNT=4)
#    - Other fixed environment variables for uvicorn.
ENV WORKER_COUNT=${WORKER_COUNT:-1}
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV APP_MODULE="slm_server.app:app"

# 3. Set Working Directory
WORKDIR /app

# 4. Install uv - the project's package manager
RUN pip install uv

# 5. Copy project dependency definitions
COPY pyproject.toml uv.lock ./

# 6. Install dependencies using uv
#    --system installs into the main python environment, not a venv
RUN uv pip install --system .

# 7. Copy the application code
COPY ./slm_server ./slm_server

# 8. Declare the models volume
#    This marks `/app/models` as a mount point for users.
#    By default, the user should mount their local `./models` directory here.
#    e.g., `docker run -v ./models:/app/models ...`
VOLUME /app/models

# 9. Expose the application port
EXPOSE 8000

# 10. Define the command to run the application
#     Using shell form to allow the WORKER_COUNT environment variable to be substituted.
CMD uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT} --workers ${WORKER_COUNT}
