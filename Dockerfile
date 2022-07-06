FROM nvidia/cuda:9.0-base
CMD nvidia-smi


FROM continuumio/anaconda3
WORKDIR /naturident

COPY ./* .

RUN conda env create -f condaEnvironment.yml

RUN conda init bash
RUN echo "conda activate naturidentOnPallets" > ~/.bashrc
RUN pip install -r requirements.txt
RUN conda install -c conda-forge faiss-gpu

RUN sh dataset-download.sh
RUN python test-reid.py
