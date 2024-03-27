# Official Python image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR  /app/api

# Copy only the necessary files for the API service
COPY main.py .
COPY best_model_2023-07-28.pkl .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the FastAPI server will run
EXPOSE 8000

# Run the FastAPI server when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
