# Enter the path to the signature embedding model and dataset here
ML_MODEL_NAME = "model.pth.tar"
MODEL_DIR_PATH = "./models/"

DATASET_NAME = "pallet-block-32965" # ids5000 or pallet-block-98382-dataset1b or arvato-15200ids
DATASET_PATH ={
    "pallet-block-502_HD_jpeg_C_RR_RL": "Datasets/pallet-block-502_HD_jpeg_C_RR_RL",
    "ids5000":  "./se-3951/ids5000.zip",
    "pallet-block-98382-dataset1b": "/home/naturident/datasets/pallet-block-98382-dataset1b/bosch-pfts",
    "arvato-15200ids": "/home/naturident/datasets/arvato-15200ids/arvato-15200ids",
    "pallet-block-32965": "Datasets/pallet-block-32965/"}



import zipfile
import os
import glob
import random
import numpy as np
import pandas as pd
import faiss
import torch
import subprocess
from tqdm import tqdm
from feature_extractor import FeatureExtractor
from PIL import Image

res = None
if torch.cuda.is_available():
    print("Using CUDA GPU")
    res = faiss.StandardGpuResources()
else:
    print("Using CPU")

model_path = MODEL_DIR_PATH + ML_MODEL_NAME
model_name = 'pcb_p4'

extractor = FeatureExtractor(
    model_name=model_name,
    model_path=model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def extract_vectors(pf_images: [np.ndarray]) -> [np.ndarray]:
    """
    Extracts feature vectors with personReid from pallet feet images
    :param pf_images: list of pallet feet images
    :return: list of feature vector
    """
    # extract vector with ML
    db_vectors = extractor(pf_images)

    assert db_vectors.shape[0] == len(pf_images), "there should be one vector for each pallet feet"

    db_vectors /= np.linalg.norm(db_vectors, axis=1)[:, np.newaxis]  # normalize vectors before we add them to the index
    vectors = db_vectors.tolist()
    return vectors

def unzip_dataset():
    with zipfile.ZipFile(DATASET_PATH[DATASET_NAME], "r") as zip_ref:
        zip_ref.extractall(DATASET_PATH[DATASET_NAME])
        DATASET_PATH[DATASET_NAME] = "se-3951/dataset/" + DATASET_NAME

if DATASET_PATH[DATASET_NAME].endswith(".zip"):
    unzip_dataset()

class PalletFeetImage:
    def __init__(self, img_file_path: str):
        self.img_file_path = img_file_path
        filename = os.path.basename(img_file_path)
        split = filename.split('.')[0].split('_')
        self.pf_id = int(split[1][3:])
        self.cam_id = int(split[3])
        self.frame_ctr = int(split[4])

    def get_np_image(self) -> np.ndarray:
        with Image.open(self.img_file_path) as img:
            np_img = np.array(img)
        return np_img

    def set_img_ctr(self, img_ctr: int):
        self.img_ctr = img_ctr

    def save_signature(self, signature: np.ndarray):
        self.signature = signature.numpy()

    def as_dict(self):
        return {'pf_id': self.pf_id, 'cam_id': self.cam_id, 'frame_ctr': self.frame_ctr, 'img_file_path': self.img_file_path, 'signature': self.signature}

    def __repr__(self):
        return repr(f"PalletBlock: (pf_id: {self.pf_id}, cam_id: {self.cam_id}, frame_ctr: {self.frame_ctr})")


if DATASET_NAME == "dataset2-ids32965":
    files = glob.glob(DATASET_PATH[DATASET_NAME] + "/*/*/*.jpg")
elif DATASET_NAME == "pallet-block-502_HD_jpeg_C_RR_RL":
    files = glob.glob(DATASET_PATH[DATASET_NAME] + "/*/*/*/*.jpeg")

pf_images = []
for file in files:
    pf_images.append(PalletFeetImage(file))
print("Images found: ", len(pf_images))

pf_images = list(sorted(pf_images, key=lambda pfi: pfi.pf_id))
#print(pf_images[:8])

if not os.path.exists(DATASET_NAME+model_name+"-signatures.pickle"):
    batchSize = 50
    with tqdm(total=len(pf_images)) as progress_bar:
        for idx in range(0, len(pf_images), batchSize):
            pf_images_batch = pf_images[idx:idx+batchSize]
            np_imgs = list(map(lambda pf_image: pf_image.get_np_image(), pf_images_batch))
            signatures = extractor(np_imgs)
            for batch_idx in range(len(pf_images_batch)):
                pf_images_batch[batch_idx].save_signature(signatures[batch_idx].cpu())
            progress_bar.update(batchSize) # update progress
    df = pd.DataFrame([x.as_dict() for x in pf_images])
    df['frame_idx'] = -1
    for (pf_id, cam_id), group in tqdm(df.groupby(['pf_id', 'cam_id'])):
        rows = df.loc[list(group.index)].sort_values(['pf_id', 'cam_id', 'frame_ctr']).reset_index()
        if(len(group.index)) < 2:
            continue
        for i in range(len(group.index)):
            df.loc[[group.index[i]], 'frame_idx']  = rows.index[rows['frame_ctr'] == rows.iloc[i]['frame_ctr']].tolist()[0]
    df.drop(df[df['frame_idx']==-1].index, inplace=True)
    #print(df.head(), df.describe())
    # save dataframe such that we can restore things from here
    df.to_pickle(DATASET_NAME+model_name+"-signatures.pickle")

df = pd.read_pickle(DATASET_NAME+model_name+"-signatures.pickle")
df['cam_id'] = pd.Categorical(df['cam_id'] )
df.info()
db_vectors = np.array(df["signature"].to_list())
db_vector_ids = np.array(df["pf_id"].to_list())
dimension = db_vectors.shape[1]
#print(type(db_vectors), db_vectors.shape)
db_vectors /= db_vectors.sum(axis=1)[:,np.newaxis] # normalize vectors before we add them to the index

def new_ip_index(dimension):
    index_flat_ip = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))  # inner product
    if res:
        index_flat_ip = faiss.index_cpu_to_gpu(res, 0, index_flat_ip)
    return index_flat_ip

def train_index(index, vectors, ids):
  print("Trained:", index.is_trained)   # False
  if not index.is_trained:
    index.train(vectors[:(len(vectors))])  # train on the database vectors
    print("Trained:", index.is_trained)  # True
  print("Vectors in Index:", index.ntotal)   # 0
  index.add_with_ids(vectors, ids)   # add the vectors and update the index
  print("Trained:", index.is_trained)  # True
  print("Vectors in Index:", index.ntotal)   # 200

def train_index_from_rows(index, rows: pd.DataFrame):
    assert rows['pf_id'].is_unique, "Expected unique pallet feet ids"
    vectors = np.array(rows["signature"].to_list())
    ids = np.array(rows["pf_id"].to_list()).astype('int64')
    vectors /= vectors.sum(axis=1)[:,np.newaxis] # normalize vectors before we add them to the index
    train_index(index, vectors, ids)

def get_test_and_train_rows():
    if DATASET_NAME == "ids5000" or DATASET_NAME == "arvato-15200ids" or DATASET_NAME == "dataset2-ids32965":
        fst_frames = df.sort_values(['pf_id', 'cam_id', 'frame_idx']).groupby(['pf_id', 'cam_id']).head(1)

        train_rows = fst_frames.loc[fst_frames['cam_id'] == 1]
        test_rows = fst_frames.loc[fst_frames['cam_id'] == 2]
    elif DATASET_NAME == "pallet-block-98382-dataset1b":
        fst_frames = df.sort_values(['pf_id', 'frame_idx', 'cam_id'])

        train_rows = fst_frames.loc[fst_frames['frame_idx'] == 0]
        test_rows = fst_frames.loc[fst_frames['frame_idx'] == 1]
    else:
        raise Exception("Not implemented Dataset")
    return train_rows, test_rows


train_rows, test_rows = get_test_and_train_rows()
test_vectors = np.array(test_rows['signature'].to_list())
test_vectors /= test_vectors.sum(axis=1)[:,np.newaxis] # normalize vectors before we add them to the index
dimension = test_vectors.shape[1]
index = new_ip_index(dimension)
train_index_from_rows(index, train_rows)
result = pd.DataFrame(test_rows)
k=2
distances, indices = index.search(test_vectors, k)

result['nn_pf_id'], result['2ndnn_pf_id'] = indices.transpose(1,0)
result['nn_distance'], result['2ndnn_distance'] = distances.transpose(1,0)
result['nn_match'] = result['pf_id'] == result['nn_pf_id']
result['2ndnn_match'] = result['pf_id'] == result['2ndnn_pf_id']

print("Correct as 1st nearest neighbour", len(result[result['nn_match']])/len(result))
print("Correct as 2nd nearest neighbour", len(result[result['2ndnn_match']])/len(result))

