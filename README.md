# naturident
Application of re-identification on EPAL pallet block images
- Re-identification is performed by training the PCB network on the pallet images and using the network to extract feature vectors for each pallet block.
- An faiss index is trained based on the features extracted for a train set with some pallet id's. The test-set is the images of the same pallet-id's but with different camera. We search the nearest neighbour in the index for a feature vector of a pallet-id from the test-set. We check to see if the feature vector of pallet-id from test-set is closer to the pallet-id from train-set thus reidentifying the same pallet-id from two different cameras.

## Testing/Re-identifying images
- run the python script "test-reid.py" with possible options to perform the re-identification.

--model_path,help="Path to folder containing model weights",default='models/'<br />
--dataset_path,help="Path to folder containing the dataset",default='Datasets/pallet-block-32965'<br />
--dataset_name,help="Name of the dataset (This can be used to )",default="pallet-block-32965"<br />
--model-name,help='Name of the model used as a feature extractor',default='pcb_p4'<br />
--frac-test,help="To select fraction of id's in the test set for re-identifying", default=1,type=float<br />

Or, put the pallet-block-32965 dataset and the trained model in the corresponding folders and run

```console
    python test-reid.py
```

Alternatively run using the docker file
- map your local folder as a volume in the docker-compose.yml by changing the D:/pallet-feet-32965 to a folder of your local and run
```console
    docker-compose up
```
