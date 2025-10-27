# Use the official Python 3.10.13 base image
FROM python:3.10.13-slim

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Command to run the app
CMD ["python", "app.py"]
