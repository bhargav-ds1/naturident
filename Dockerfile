FROM nvidia/cuda:9.0-base
CMD nvidia-smi


FROM continuumio/anaconda3
WORKDIR /naturident
COPY . .
RUN pip install -r requirements.txt
RUN conda install -c conda-forge faiss-gpu

RUN conda env create -f condaEnvironment.yml && conda init bash && echo "conda activate naturidentOnPallets" > ~/.bashrc
#RUN python test-reid.py
