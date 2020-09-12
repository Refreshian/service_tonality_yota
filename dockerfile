FROM python:3.6-slim

COPY . /root

WORKDIR /root

# ADD requirements.txt
RUN pip install --default-timeout=2000 -r requirements.txt