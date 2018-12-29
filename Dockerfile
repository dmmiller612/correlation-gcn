FROM conda/miniconda3

RUN apt-get update && apt-get install -y build-essential \
    libxrender-dev \
    libxext6 \
    libsm6

RUN mkdir -p /opt/gcn
COPY . /opt/gcn
WORKDIR /opt/gcn

RUN conda create -n gcn python=3.6 \
    && conda env update -n gcn -f environment.yml \
    && echo "source activate gcn" >> ~/.bashrc

