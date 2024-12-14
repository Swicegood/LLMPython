
# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Pre-set the timezone to avoid prompts
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

    
# Set the working directory
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir flask==3.0.3 transformers==4.44.2 pillow requests bitsandbytes==0.43.0 accelerate==0.34.0 wheel

# Make port 1234 available to the world outside this container
EXPOSE 1234

# Copy the current directory contents into the container at /app
COPY . /app

# Run app.py when the container launches
CMD ["python", "app.py"]