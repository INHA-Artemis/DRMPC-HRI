FROM continuumio/miniconda3:24.1.2-0

WORKDIR /opt/drmpc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/drmpc

RUN conda env create -f environment.yml && conda clean -afy

RUN git clone https://github.com/sybrenstuvel/Python-RVO2.git /opt/Python-RVO2 && \
    /opt/conda/bin/conda run -n social_navigation bash -lc "cd /opt/Python-RVO2 && python setup.py build && python setup.py install"

RUN git clone https://github.com/utiasASRL/pysteam.git /opt/pysteam

ENV PYTHONPATH=/opt/drmpc:/opt/pysteam

CMD ["bash"]
