# Use 11.8 for PyTorch
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV TZ=Asia/Tokyo

RUN apt-get update
RUN apt-get install -y git git-lfs

RUN apt-get install -y curl
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda3.sh
RUN bash miniconda3.sh -b -p /opt/miniconda3
RUN rm miniconda3.sh
ENV PATH /opt/miniconda3/bin:$PATH
RUN conda init bash

# Ref: https://pytorch.org/get-started/locally/
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN rm requirements.txt
