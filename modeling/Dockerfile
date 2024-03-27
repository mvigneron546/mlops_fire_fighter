# Official Python image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app/modeling

# Copy only the necessary files for the modeling service
COPY main.py .
COPY Dockerfile .
COPY requirements.txt .
COPY utils_modeling /app/modeling/utils_modeling

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to run the main.py script
ENTRYPOINT ["python", "main.py"]
