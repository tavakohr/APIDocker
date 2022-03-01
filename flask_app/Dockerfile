FROM python:3.6.7

WORKDIR usr/src/flask_app
RUN apt-get update
RUN pip install --upgrade pip setuptools wheel
RUN pip install opencv-python==4.5.5.62
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pickle-mixin

COPY . .