FROM nvidia/cuda:9.0-base
CMD nvidia-smi


FROM continuumio/anaconda3
WORKDIR /naturident

COPY condaEnvironment.yml ./
COPY requirements.txt ./


RUN conda env create -f condaEnvironment.yml

RUN conda init bash
RUN echo "conda activate naturidentOnPallets" > ~/.bashrc
RUN pip install -r requirements.txt
RUN conda install -c conda-forge faiss-gpu

# download dataset from owncloud
#wget -O se-3951/ids5000.zip https://owncloud.fraunhofer.de/index.php/s/Wdt7MJgMDwFcLK1/download
