# Base Image
FROM continuumio/miniconda3:main

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first for better caching
COPY requirements.txt /app/
COPY .env.example /app/.env

# Install dependencies using pip and clean up after
RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code into the container
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]