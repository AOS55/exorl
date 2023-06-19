FROM ubuntu:22.04

COPY . src

WORKDIR /src

ENV PORT=8080

# setup env
RUN apt update -y
RUN apt-get update && apt-get install -y \
    software-properties-common

RUN add-apt-repository universe

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    swig \
    cmake \
    libjpeg-dev \
    zlib1g-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libgl1-mesa-glx \
    libosmesa6

RUN alias python=python3

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip3 install ./libraries/gym

RUN export DISPLAY=:0.0

CMD bash

# CMD ["python3", "src/test_envgen.py"]
