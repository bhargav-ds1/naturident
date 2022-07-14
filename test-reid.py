import random
import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from feature_extractor import FeatureExtractor
from PIL import Image

parser = argparse.ArgumentParser(description='Arguments to test the re-identification')
parser.add_argument('--model-path',help="Path to folder containing model weights",default='models/')
parser.add_argument('--dataset-path',help="Path to folder containing the dataset",default='Datasets/pallet-block-32965')
parser.add_argument('--dataset-name',help="Name of the dataset (This can be used to )",default="pallet-block-32965")
parser.add_argument('--model-name',help='Name of the model used as a feature extractor',default='pcb_p4')
parser.add_argument('--frac-test',help="To select fraction of id's in the test set for re-identifying", default=1,type=float)
args = parser.parse_args()
res = None

files = []
for ext in ['jpg','jpeg']:
    files.extend(glob.glob(args.dataset_path+"/**/*."+ext,recursive=True))

assert len(files) == len(list(filter(lambda a: re.match('\w{2}\_\d+\_cam\_\d+\_.*\d+',os.path.basename(a)),files))), "" \
    "Image names should be defined similar to the images from pallet-block-32965 e.g(pf_1001000000000033906_cam_2_1990.jpg)"

if torch.cuda.is_available():
    print("Using CUDA GPU")
    res = faiss.StandardGpuResources()
else:
    print("Using CPU- Re-identifying can be slow.")

extractor = FeatureExtractor(
    model_name=args.model_name,
    model_path=args.model_path+'model.pth.tar',
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

if len(files) == 0 and not os.path.exists(args.dataset_path+'/'+args.dataset_name+args.model_name+"-signatures.pickle"):
    print('Could not find any images from the Dataset. Check if the dataset is part of the Datasets folder or use --dataset-path to specify a dataset path.')
    exit(0)
if len(files)!=0 and not os.path.exists(args.dataset_path+'/'+args.dataset_name+args.model_name+"-signatures.pickle"):
    print('Extracting features/signatures from the images and saving as pickle file.')
    pf_images = []
    for file in files:
        pf_images.append(PalletFeetImage(file))
    print("Images found: ", len(pf_images))
    pf_images = list(sorted(pf_images, key=lambda pfi: pfi.pf_id))
    #print(pf_images[:8])
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
    df.to_pickle(args.dataset_path+'/'+args.dataset_name+args.model_name+"-signatures.pickle")
    print(f'Saving extracted feature/signatures from images as {args.dataset_name+args.model_name+"-signatures.pickle"}')

print(f'Reading signatures from {args.dataset_path+"/"+args.dataset_name+args.model_name+"-signatures.pickle"}')
df = pd.read_pickle(args.dataset_path+'/'+args.dataset_name+args.model_name+"-signatures.pickle")
print(f'Shape of loaded signature pickle file {df.shape}')
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
    print(f"Training the index using {len(ids)} feature vectors..")
    train_index(index, vectors, ids)


def get_test_and_train_rows():
    fst_frames = df.sort_values(['pf_id', 'cam_id', 'frame_idx']).groupby(['pf_id', 'cam_id']).head(1)
    train_rows = fst_frames.loc[fst_frames['cam_id'] == 1]
    test_rows = fst_frames.loc[fst_frames['cam_id'] == 2]
    return train_rows, test_rows


train_rows, test_rows = get_test_and_train_rows()
test_rows = test_rows.head(int(test_rows.shape[0]*args.frac_test))
test_vectors = np.array(test_rows['signature'].to_list())
test_vectors /= test_vectors.sum(axis=1)[:,np.newaxis] # normalize vectors before we add them to the index
dimension = test_vectors.shape[1]
index = new_ip_index(dimension)
train_index_from_rows(index, train_rows)
result = pd.DataFrame(test_rows)
k=2
print(f"Searching/Re-identifying {test_rows.shape[0]} id's and getting k={k} neighbours (source faiss)...")
distances, indices = index.search(test_vectors, k)

result['nn_pf_id'], result['2ndnn_pf_id'] = indices.transpose(1,0)
result['nn_distance'], result['2ndnn_distance'] = distances.transpose(1,0)
result['nn_match'] = result['pf_id'] == result['nn_pf_id']
result['2ndnn_match'] = result['pf_id'] == result['2ndnn_pf_id']

print("Correct as 1st nearest neighbour", len(result[result['nn_match']])/len(result))
print("Correct as 2nd nearest neighbour", len(result[result['2ndnn_match']])/len(result))

#------------------
n = 6
train_rows, test_rows = get_test_and_train_rows()
skip_rows = len(train_rows) % n
if skip_rows > 0:
    print(f"skipping first {skip_rows} rows")
train_rows = pd.DataFrame(train_rows[skip_rows:])
test_rows = pd.DataFrame(test_rows[skip_rows:])

# randomly assign each row to a pallet
random.seed(42)
permutation = random.sample(list(range(int(len(train_rows)/n)))*n, k=len(train_rows))
train_rows['pallet'] = permutation
test_rows['pallet'] = permutation


test_vectors = np.array(test_rows['signature'].to_list())
test_vectors /= test_vectors.sum(axis=1)[:,np.newaxis] # normalize vectors before we add them to the index
dimension = test_vectors.shape[1]

index = new_ip_index(dimension)
train_index_from_rows(index, train_rows)

train_rows, test_rows = get_test_and_train_rows()


skip_rows = len(train_rows) % n
if skip_rows > 0:
    print(f"skipping first {skip_rows} rows")
train_rows = pd.DataFrame(train_rows[skip_rows:])
test_rows = pd.DataFrame(test_rows[skip_rows:])

# randomly assign each row to a pallet
random.seed(42)
permutation = random.sample(list(range(int(len(train_rows)/n)))*n, k=len(train_rows))
train_rows['pallet'] = permutation
test_rows['pallet'] = permutation


test_vectors = np.array(test_rows['signature'].to_list())
test_vectors /= test_vectors.sum(axis=1)[:,np.newaxis] # normalize vectors before we add them to the index
dimension = test_vectors.shape[1]

index = new_ip_index(dimension)
train_index_from_rows(index, train_rows)
result = pd.DataFrame(test_rows)

k=2
distances, indices = index.search(test_vectors, k)

result['nn_pf_id'], result['2ndnn_pf_id'] = indices.transpose(1,0)
result['nn_similarity'], result['2ndnn_similarity'] = distances.transpose(1,0)
result['nn_match'] = result['pf_id'] == result['nn_pf_id']
result['2ndnn_match'] = result['pf_id'] == result['2ndnn_pf_id']
result['nn_pallet'] = result['nn_pf_id'].apply(lambda pfid: train_rows[train_rows['pf_id'] == pfid]['pallet'].iat[0]).astype(int)
result['2ndnn_pallet'] = result['2ndnn_pf_id'].apply(lambda pfid: train_rows[train_rows['pf_id'] == pfid]['pallet'].iat[0]).astype(int)

def weighted_agg(group_df: pd.DataFrame):
    pallets = {}
    for row in group_df.itertuples():
        if row.nn_pallet not in pallets:
            pallets[row.nn_pallet] = 0
        pallets[row.nn_pallet] += row.nn_similarity

        if row[3] not in pallets:
            pallets[row[3]] = 0
        pallets[row[3]] += row[4]
    return max(pallets, key=pallets.get)


nn_majority_vote_pallet = result.groupby(['pallet'])['nn_pallet'].agg(lambda x:x.value_counts().index[0])
weighted_2nn_pallet = result.groupby('pallet')[['nn_pallet', 'nn_similarity', '2ndnn_pallet', '2ndnn_similarity']].apply(weighted_agg)


pallets_identified_as = pd.concat([nn_majority_vote_pallet, weighted_2nn_pallet], axis=1, keys=["NN_Majority_Vote", "Weighted_2NN"])

pallets_identified_as['majority_vote_match'] = pallets_identified_as.index == pallets_identified_as['NN_Majority_Vote']
pallets_identified_as['wighted_2nn_match'] =  pallets_identified_as.index == pallets_identified_as['Weighted_2NN']

print("Majority Vote", len(pallets_identified_as[pallets_identified_as['majority_vote_match']])/len(pallets_identified_as))
print("Weighted", len(pallets_identified_as[pallets_identified_as['wighted_2nn_match']])/len(pallets_identified_as))
