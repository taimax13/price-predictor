# Stage 1: Train
FROM python:3.8-slim AS train

WORKDIR /app

# Copy the application files
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Run the training script
# Ensure that your main function is correctly set up to be called
RUN python modelTrainer.py

# Stage 2: Pack and Serve
FROM python:3.8-slim AS pack-serving

WORKDIR /app

# Copy the necessary files from the train stage
COPY --from=train /app /app

# Install runtime dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5001

# Start the Flask application
CMD ["python", "app.py"]
