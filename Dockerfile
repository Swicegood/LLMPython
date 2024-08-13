
# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Pre-set the timezone to avoid prompts
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

    
# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir flask transformers pillow requests bitsandbytes accelerate wheel

# Make port 1234 available to the world outside this container
EXPOSE 1234

# Run app.py when the container launches
CMD ["python", "app.py"]