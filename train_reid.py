from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import os
import os.path as osp
import time

import mlflow
import numpy as np
import torch
from PIL import Image
from mlflow.tracking import MlflowClient
from tensorflow.python.summary.summary_iterator import summary_iterator
import torchreid
from torchreid import data
from torchreid.data import ImageDataset
from torchreid.utils import FeatureExtractor, read_image
from tqdm import tqdm

from detection import detection_pipeline
from detection.utils.distance_measurements import DistanceMeasurement
from detection.utils.log_writer import LogWriter

save_path = '../tests/log/reid'

dataset_names = [
    # 'dataset/',
    # 'Bosch/',
    # 'mobile_data/'
    #'/home/naturident/datasets/industrialImages/',
    #'/home/naturident/datasets/pallet-block-98382-dataset1b/',
    #'/home/naturident/datasets/ids5000/',
    #'/home/naturident/datasets/arvato-15200ids/',
    '/home/naturident/datasets/arvato/',
]


class PalletDataset(ImageDataset):

    def __init__(self, root='', val=False, **kwargs):
        self.id_dict = {}
        self.id_count = 0
        self.root = osp.abspath(osp.expanduser(root))

        train = []
        query = []
        gallery = []

        _train = []
        _query = []
        _gallery = []

        # iterate through the datasets
        for dataset_name in dataset_names:

            self.dataset_dir = osp.join(self.root, dataset_name)
            print(dataset_name)
            _, directories, _ = next(os.walk(self.dataset_dir))
            print(directories)
            # Each list contains tuples of (img_path, pid, camid),
            # where
            # - img_path (str): absolute path to an image.
            # - pid (int): person ID, e.g. 0, 1.
            # - camid (int): camera ID, e.g. 0, 1.
            id_in_dateset = {}
            id_count_in_dataset = 0
            _ids = []
            cams =[] 
            id_offset = len(self.id_dict.keys())
            for cam in range(len(directories)):
                _, _, file_names = next(os.walk(os.path.join(self.dataset_dir, directories[cam])))

                for file_name in file_names:
                    split = file_name.split("_")
                    pallet_id = int(split[-4])
                    if pallet_id not in _ids:
                        _ids.append(pallet_id)
                        id_count_in_dataset += 1
                        if file_name.split("_")[-2] not in cams:
                            cams.append(file_name.split("_")[-2])
            print("IDs in dataset ", id_count_in_dataset)
            
            for cam in range(len(directories)):
                # ToDo: remove
                # if str(directories[cam]) != "IDs18128-35912":
                #     print('{}skipped'.format(cam))
                #     continue
                _, _, file_names = next(os.walk(os.path.join(self.dataset_dir, directories[cam])))
                #print("file_names", len(file_names))
                
                for file_name in file_names:
                    # industy dataset (bosch: pf_1011000000000008139_cam_11_431_1.jpg)
                    # Arvarto 5000ids:() pf_1001000000000005473_cam_1_676.jpg)
                    #print(file_name)
                    if file_name.split("_")[0].startswith("pf"):
                        
                        path_to_file = osp.join(self.dataset_dir, directories[cam], file_name)
                        split = file_name.split("_")
                        pallet_id = int(split[-4])
                        #print("pallet-id: ", pallet_id)

                        if pallet_id not in id_in_dateset.keys():
                            id_in_dateset[pallet_id] = self.id_count
                            pallet_id = id_in_dateset[pallet_id]
                        else:
                            pallet_id = id_in_dateset[pallet_id]

                        if pallet_id not in self.id_dict:
                            self.id_dict[pallet_id] = self.id_count
                            self.id_count += 1
                            pallet_id = self.id_dict[pallet_id]
                        else:
                            pallet_id = self.id_dict[pallet_id]

                        #print("cam-3: ", file_name.split("_")[-3])
                        #print("ID-Count: ", self.id_count )
                        # split lighton and lightoff to different cams
                        _cam = 1 if file_name.split("_")[-2] == '1' else 0
                        #print(path_to_file)
                        #print(pallet_id)
                        #print(_cam)
                        if pallet_id - id_offset < id_count_in_dataset * 0.1: # 0.8
                            _train.append((path_to_file, pallet_id, int(_cam)))
                        elif pallet_id - id_offset < id_count_in_dataset * 0.2: # 0.9
                            _query.append((path_to_file, pallet_id, int(_cam)))
                        if val:
                            if pallet_id - id_offset >= id_count_in_dataset * 0.2: # 0.9
                                _gallery.append((path_to_file, pallet_id, int(_cam)))
                        else:
                            if id_count_in_dataset * 0.1 <= pallet_id - id_offset < id_count_in_dataset * 0.2: # 0.8 - 0.9
                                _gallery.append((path_to_file, pallet_id, int(_cam)))

                    # laboratory dataset
                    else:
                        path_to_file = osp.join(self.dataset_dir, directories[cam], file_name)
                        split = file_name.split("_")
                        pallet_id = int(split[-2])+id_offset
                        _cam = cam

                        if pallet_id not in id_in_dateset.keys():
                            id_in_dateset[pallet_id] = self.id_count
                            pallet_id = id_in_dateset[pallet_id]
                        else:
                            pallet_id = id_in_dateset[pallet_id]

                        if pallet_id not in self.id_dict:
                            self.id_dict[pallet_id] = self.id_count
                            self.id_count = self.id_count + 1
                            pallet_id = self.id_dict[pallet_id]
                        else:
                            pallet_id = self.id_dict[pallet_id]

                        if pallet_id - id_offset < id_count_in_dataset * 0.8:
                            _train.append((path_to_file, pallet_id, int(_cam)))
                        elif pallet_id - id_offset < id_count_in_dataset * 0.9:
                            _query.append((path_to_file, pallet_id, int(_cam)))
                        if val:
                            if pallet_id - id_offset >= id_count_in_dataset * 0.9:
                                _gallery.append((path_to_file, pallet_id, int(_cam)))
                        else:
                            if id_count_in_dataset * 0.8 <= pallet_id - id_offset < id_count_in_dataset * 0.9:
                                _gallery.append((path_to_file, pallet_id, int(_cam)))

        # for path_to_file, pallet_id, cam in _query:



        #shift ids
        shifted_ids = 0
        id_in_set = {}
        for path_to_file, pallet_id, cam in _train:
            if pallet_id not in id_in_set.keys():
                id_in_set[pallet_id] = shifted_ids
                shifted_ids += 1
                pallet_id = id_in_set[pallet_id]
            else:
                pallet_id = id_in_set[pallet_id]
            train.append((path_to_file, pallet_id, cam))

        for path_to_file, pallet_id, cam in _query:
            if pallet_id not in id_in_set.keys():
                id_in_set[pallet_id] = shifted_ids
                shifted_ids += 1
                pallet_id = id_in_set[pallet_id]
            else:
                pallet_id = id_in_set[pallet_id]
            query.append((path_to_file, pallet_id, cam))

        for path_to_file, pallet_id, cam in _gallery:
            if pallet_id not in id_in_set.keys():
                id_in_set[pallet_id] = shifted_ids
                shifted_ids += 1
                pallet_id = id_in_set[pallet_id]
            else:
                pallet_id = id_in_set[pallet_id]
            gallery.append((path_to_file, pallet_id, cam))
        
        if not val:
            gallery.clear()
            gallery.extend(query)


        super(PalletDataset, self).__init__(train, query, gallery, **kwargs)



class PersonReId:

    def __init__(self, backbone='pcb_p4', model_path=None, run_testcases=False):
        self.model_path = model_path
        self.backbone = backbone
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        if model_path:
            self.extractor = FeatureExtractor(
                model_name=self.backbone,
                model_path= model_path,
                device=self.device
            )

        self.epochs = 60
        self.optim = 'sgd'
        #self.optim = 'adam'
        self.lr = 0.1
        self.staged_lr = True # prev. True, default = False
        self.base_lr_mult = 0.5
        #self.weight_decay = 5e-04 # defalut value 5e-04
        #self.momentum = 0.9
        self.height = 384
        self.width = 128
        #self.height = 192
        #self.width = 64
        print("Image input-size: ["+str(self.width) +", " + str(self.height) + "]")
        
        # scheduler single_step
        #self.lr_scheduler='single_step'
        #self.stepsize = 15 # prev. 30
        #self.gamma = 1

        # scheduler multi_step
        self.lr_scheduler = 'multi_step'
        self.stepsize = [20, 25, 30, 35, 40, 45, 50, 55]
        self.gamma = 0.7

        self.model_uri = ""
        self.run_testcases = run_testcases
        self.loss = ""
        self.experiment_name = 'reid {}'.format(self.backbone)
        self.multiplier = 'lr_mult:{}'.format(self.base_lr_mult) if self.staged_lr else ''
        # run_name = '{}x{} epochs:{} optim:{} lr:{} {}'.format(self.height, self.width, self.epochs, self.optim, self.lr, multiplier )
        self.run_name = '{}x{}'.format(self.height, self.width)
        LogWriter.set_backone(self.backbone)
        LogWriter.set_log_path(save_path)

    def train(self):
        LogWriter.set_log_path(self.backbone)

        mlflow.set_experiment(experiment_name=self.experiment_name)
        with mlflow.start_run(run_name='train {}'.format(self.run_name)) as run:
            LogWriter.set_experiment_name('train {}'.format(self.run_name))
            start_time = time.perf_counter()

            torchreid.data.register_image_dataset('pallets', PalletDataset)

            datamanager = torchreid.data.ImageDataManager(
                root='../tests',
                sources='pallets',
                targets='pallets',
                height=self.height,
                width=self.width,
                batch_size_train=20,
                batch_size_test=20,
                # transforms=['random_flip', 'random_crop']
                transforms=['random_crop']
            )

            model = torchreid.models.build_model(
                name=self.backbone,
                num_classes=datamanager.num_train_pids,
                loss='softmax', # triplet
                pretrained=True
            )

            if self.device == 'cuda':
                model = model.cuda()

            optimizer = torchreid.optim.build_optimizer(
                model,
                optim=self.optim,
                lr=self.lr,
                staged_lr=self.staged_lr,
                base_lr_mult=self.base_lr_mult,
                #weight_decay=self.weight_decay,
                #momentum=self.momentum
            )

            scheduler = torchreid.optim.build_lr_scheduler(
                optimizer,
                # stepsize=self.stepsize,
                # lr_scheduler='single_step',
                lr_scheduler=self.lr_scheduler,
                stepsize=self.stepsize,
                gamma=self.gamma,
                
            )

            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                label_smooth=True
            )

            # engine = torchreid.engine.ImageTripletEngine(
            #     datamanager,
            #     model,
            #     optimizer=optimizer,
            #     scheduler=scheduler,
            #     label_smooth=True
            # )

            engine.run(
                save_dir=os.path.join(save_path, self.backbone),
                max_epoch=self.epochs,
                eval_freq=10,
                print_freq=100,
                test_only=False,
                mlflow=mlflow
            )

            # get Training loss from log
            print()
            _dir = os.path.join(save_path, self.backbone)
            files = os.listdir(_dir)
            paths = [os.path.join(_dir, basename) for basename in files]
            log_path = max(paths, key=os.path.getctime)
            max_iter = 0
            for e in summary_iterator(log_path):
                if e.step >= max_iter:
                    for v in e.summary.value:
                        if v.tag == 'Train/loss':
                            self.loss = v.simple_value

            # measure and track training time
            end_time = time.perf_counter()
            elapsed_time = '{:.2f} s'.format(end_time - start_time)

            # track learning parameters
            mlflow.log_param('epochs', self.epochs)
            mlflow.log_param('optim', self.optim)
            mlflow.log_param('lr', self.lr)
            mlflow.log_param('staged_lr', self.staged_lr)
            mlflow.log_param('stepsize',  self.stepsize)
            mlflow.log_param('base_lr_mult', self.base_lr_mult)
            mlflow.log_param('image height', self.height)
            mlflow.log_param('image width', self.width)
            mlflow.log_param('loss', self.loss)
            mlflow.log_param('computation time', elapsed_time)

            # save model
            mlflow.pytorch.log_model(model, 'model')

            self.print_logged_info(mlflow.get_run(run_id=run.info.run_id))

            self.model_uri = "runs:/{}/model".format(run.info.run_id)

    def run_simple_tests_single_cam(self, reg_frame="1"):
        mlflow.set_experiment(experiment_name=self.experiment_name)
        for dist_metric in DistanceMeasurement.available_distances:
            with mlflow.start_run(run_name='{} {}'.format(dist_metric, self.run_name)) as run:
                LogWriter.set_experiment_name('{} {}'.format(dist_metric, self.run_name))
                start_time = time.perf_counter()

                mlflow.log_param('epochs', self.epochs)
                mlflow.log_param('optim', self.optim)
                mlflow.log_param('lr', self.lr)
                mlflow.log_param('staged_lr', self.staged_lr)
                mlflow.log_param('base_lr_mult', self.base_lr_mult)
                mlflow.log_param('image height', self.height)
                mlflow.log_param('image width', self.width)
                mlflow.log_param('loss', self.loss)
                mlflow.log_param('metric', dist_metric)

                _detection_pipeline = detection_pipeline.DetectionPipeline(
                    model_path_reid=os.path.join(
                        os.path.join(save_path, self.backbone),
                        'model/model.pth.tar-{}'.format(self.epochs)), #'/home/naturident/Downloads/pcb_p4_bosch.pth.tar'), #
                    vector_creator="PersonReId",
                    distance_measurement=dist_metric,
                    backbone=self.backbone
                )

                dataset = PalletDataset(
                    root='../tests',
                    val=True
                )

                images = []
                # images_train = [image[0] for image in dataset.train]
                # images.extend(images_train)
                # images_query = [image[0] for image in dataset.query]
                # images.extend(images_query)
                images_gallery = [image[0] for image in dataset.gallery]
                images.extend(images_gallery)

                images_by_frame = {}

                print('Testing with distance metric: {}'.format(dist_metric))

                ids_registered={"1":[], "2":[]}

                for filename in images:
                    split = filename.split("_")
                    cam = split[-2] # -4: remove .jpg & -2 camera_id
                    id = split[-4]
                    if id in ids_registered[cam]:
                        frame = str(int(cam) + 2)
                    else:
                        ids_registered[cam].append(id)
                        frame = cam
                    if str(frame) not in images_by_frame.keys():
                        images_by_frame[str(frame)] = []
                    images_by_frame[str(frame)].append(filename)
                print(images_by_frame.keys())
                registration_set = 0
                reidentification_set = 0

                for key in images_by_frame.keys(): #currently images_by_cam
                    if key == str(reg_frame):
                        registration_set += len(images_by_frame[key])
                        print("adding key {} with {} images to reg".format(key, len(images_by_frame[key])))
                    else:
                        reidentification_set += len(images_by_frame[key])
                        print("adding key {} with {} images to identification".format(key, len(images_by_frame[key])))

                print("")
                print("IDs for registration: {}".format(registration_set))
                print("Images for reidentification: {}".format(reidentification_set))
                print("")

                test_scenarios = {
                    "All Frames Identification": [0, 0, 0, 0, 0, 0, 0, 0],
                }
                print("")

                print("registering pallets of frame {}".format(reg_frame))
                _detection_pipeline.register_multiple_pallets(images_by_frame[reg_frame], batch_size=100)
                print("registration done")

                results = []
                for frame in images_by_frame.keys():
                    if frame != reg_frame and len(images_by_frame[frame]) > 0:
                        source = reg_frame
                        destination = frame
                        print("")
                        print("reidentificating palletfeet of frame {}...".format(frame))
                        identifications = _detection_pipeline.identify_multiple_pallets(images_by_frame[frame], batch_size=100, k=30)
                        print("reidentification of frame {} done".format(frame))


                        source_count = len(images_by_frame[source])
                        destination_count = len(images_by_frame[destination])
                        image_count = source_count if source_count <= destination_count else destination_count

                        # track results from single tests
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[1] / image_count * 100, step=1)
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[2] / image_count * 100, step=2)
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[3] / image_count * 100, step=3)
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[4] / image_count * 100, step=5)
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[5] / image_count * 100, step=10)
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[6] / image_count * 100, step=20)
                        mlflow.log_metric(key='frame{}_frame{}'.format(source, destination), value=identifications[7] / image_count * 100, step=30)
                        LogWriter.write_to_logfile('frame {} -> frame {}'.format(source, destination))
                        LogWriter.write_to_logfile(
                            'k=1: {} of {} -> {}%'.format(identifications[1], image_count, int(identifications[1] / image_count * 100)))
                        LogWriter.write_to_logfile(
                            'k=2: {} of {} -> {}%'.format(identifications[2], image_count, int(identifications[2] / image_count * 100)))
                        LogWriter.write_to_logfile(
                            'k=3: {} of {} -> {}%'.format(identifications[3], image_count, int(identifications[3] / image_count * 100)))
                        LogWriter.write_to_logfile(
                            'k=5: {} of {} -> {}%'.format(identifications[4], image_count, int(identifications[4] / image_count * 100)))
                        LogWriter.write_to_logfile(
                            'k=10: {} of {} -> {}%'.format(identifications[5], image_count, int(identifications[5] / image_count * 100)))
                        LogWriter.write_to_logfile(
                            'k=20: {} of {} -> {}%'.format(identifications[6], image_count, int(identifications[6] / image_count * 100)))
                        LogWriter.write_to_logfile(
                            'k=30: {} of {} -> {}%'.format(identifications[7], image_count, int(identifications[7] / image_count * 100)))
                        LogWriter.write_to_logfile('')
                        LogWriter.write_to_logfile('')

                        scenario="All Frames Identification"
                        test_scenarios[scenario][0] = test_scenarios[scenario][0] + image_count
                        test_scenarios[scenario][1] = test_scenarios[scenario][1] + identifications[1]
                        test_scenarios[scenario][2] = test_scenarios[scenario][2] + identifications[2]
                        test_scenarios[scenario][3] = test_scenarios[scenario][3] + identifications[3]
                        test_scenarios[scenario][4] = test_scenarios[scenario][4] + identifications[4]
                        test_scenarios[scenario][5] = test_scenarios[scenario][5] + identifications[5]
                        test_scenarios[scenario][6] = test_scenarios[scenario][6] + identifications[6]
                        test_scenarios[scenario][7] = test_scenarios[scenario][7] + identifications[7]

                # track results for test scenatios
                for test_scenario in test_scenarios.keys():
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=1)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=2)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=3)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=5)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=10)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=20)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0],
                                      step=30)

                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][1] / test_scenarios[test_scenario][
                                          0] * 100, step=1)
                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][2] / test_scenarios[test_scenario][
                                          0] * 100, step=2)
                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][3] / test_scenarios[test_scenario][
                                          0] * 100, step=3)
                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][4] / test_scenarios[test_scenario][
                                          0] * 100, step=5)
                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][5] / test_scenarios[test_scenario][
                                          0] * 100, step=10)
                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][6] / test_scenarios[test_scenario][
                                          0] * 100, step=20)
                    mlflow.log_metric(key=test_scenario,
                                      value=test_scenarios[test_scenario][7] / test_scenarios[test_scenario][
                                          0] * 100, step=30)

                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][1], step=1)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][2], step=2)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][3], step=3)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][4], step=5)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][5], step=10)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][6], step=20)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario),
                                      value=test_scenarios[test_scenario][7], step=30)

                    LogWriter.write_to_logfile(test_scenario)
                    LogWriter.write_to_logfile('k=1: {} of {} -> {}%'.format(test_scenarios[test_scenario][1],
                                                                             test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][1] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('k=2: {} of {} -> {}%'.format(test_scenarios[test_scenario][2],
                                                                             test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][2] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('k=3: {} of {} -> {}%'.format(test_scenarios[test_scenario][3],
                                                                             test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][3] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('k=5: {} of {} -> {}%'.format(test_scenarios[test_scenario][4],
                                                                             test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][4] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('k=10: {} of {} -> {}%'.format(test_scenarios[test_scenario][5],
                                                                              test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][5] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('k=20: {} of {} -> {}%'.format(test_scenarios[test_scenario][6],
                                                                              test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][6] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('k=30: {} of {} -> {}%'.format(test_scenarios[test_scenario][7],
                                                                              test_scenarios[test_scenario][0], int(
                            test_scenarios[test_scenario][7] / test_scenarios[test_scenario][0] * 100)))
                    LogWriter.write_to_logfile('')
                    LogWriter.write_to_logfile('')

                # measure and track computation time
                end_time = time.perf_counter()
                elapsed_time = '{:.2f} s'.format(end_time - start_time)
                mlflow.log_param('computation time', elapsed_time)
                self.print_logged_info(run)

    def run_tests(self):
        # run test scenarios for all available metrics
        mlflow.set_experiment(experiment_name=self.experiment_name)
        for dist_metric in DistanceMeasurement.available_distances:
            with mlflow.start_run(run_name='{} {}'.format(dist_metric, self.run_name)) as run:
                LogWriter.set_experiment_name('{} {}'.format(dist_metric, self.run_name))
                start_time = time.perf_counter()

                mlflow.log_param('epochs', self.epochs)
                mlflow.log_param('optim', self.optim)
                mlflow.log_param('lr', self.lr)
                mlflow.log_param('staged_lr', self.staged_lr)
                mlflow.log_param('base_lr_mult', self.base_lr_mult)
                mlflow.log_param('image height', self.height)
                mlflow.log_param('image width', self.width)
                mlflow.log_param('loss', self.loss)
                mlflow.log_param('metric', dist_metric)

                _detection_pipeline = detection_pipeline.DetectionPipeline(
                    model_path_reid=os.path.join(
                        os.path.join(save_path, self.backbone),
                        'model/model.pth.tar-{}'.format(self.epochs)),
                    vector_creator="PersonReId",
                    distance_measurement=dist_metric,
                    backbone=self.backbone
                )

                dataset = PalletDataset(
                    root='../tests',
                    val=True
                )
                images = [image[0] for image in dataset.gallery]

                k = 30

                images_cam_0 = {
                    "C": [],
                    "R": [],
                    "L": [],
                    "RL": [],
                    "RR": [],
                }

                images_cam_1 = {
                    "C": [],
                    "R": [],
                    "L": [],
                    "RL": [],
                    "RR": [],
                }

                cam0_prefix = ""
                for filename in images:
                    split = filename.split("_")
                    if cam0_prefix == "":
                        cam0_prefix = split[0]
                    if cam0_prefix == split[0]:
                        pallet_shot_style = split[-1][:-4]
                        images_cam_0[pallet_shot_style].append(filename)

                for filename in images:
                    split = filename.split("_")
                    if cam0_prefix != split[0]:
                        pallet_shot_style = split[-1][:-4]
                        images_cam_1[pallet_shot_style].append(filename)

                results = _detection_pipeline.run_testscenarios(images_cam_0, images_cam_1, k=k)

                test_scenarios = {
                    "Best Perspective Matching": [0, 0, 0, 0, 0, 0, 0, 0],
                    "Same Perspective Matching": [0, 0, 0, 0, 0, 0, 0, 0],
                    "Cross Matching": [0, 0, 0, 0, 0, 0, 0, 0],
                    "Full Cross Matching": [0, 0, 0, 0, 0, 0, 0, 0],
                }

                for (source, destination), values in results:

                    #possible matches
                    source_prefix = source.split('_')
                    if source_prefix[0] == 'cam0':
                        source_count = len(images_cam_0[source_prefix[1]])
                    else:
                        source_count = len(images_cam_1[source_prefix[1]])

                    destination_prefix = destination.split('_')
                    if destination_prefix[0] == 'cam0':
                        destination_count = len(images_cam_0[destination_prefix[1]])
                    else:
                        destination_count = len(images_cam_1[destination_prefix[1]])
                    image_count = source_count if source_count <= destination_count else destination_count


                    # track results from single tests
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[1]/image_count*100, step=1)
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[2]/image_count*100, step=2)
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[3]/image_count*100, step=3)
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[4]/image_count*100, step=5)
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[5]/image_count*100, step=10)
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[6]/image_count*100, step=20)
                    mlflow.log_metric(key='{}_{}'.format(source, destination), value=values[7]/image_count*100, step=30)
                    LogWriter.write_to_logfile('{} -> {}'.format(source, destination))
                    LogWriter.write_to_logfile('k=1: {} of {} -> {}%'.format(values[1], image_count, int(values[1] / image_count*100)))
                    LogWriter.write_to_logfile('k=2: {} of {} -> {}%'.format(values[2], image_count, int(values[2] / image_count*100)))
                    LogWriter.write_to_logfile('k=3: {} of {} -> {}%'.format(values[3], image_count, int(values[3] / image_count*100)))
                    LogWriter.write_to_logfile('k=5: {} of {} -> {}%'.format(values[4], image_count, int(values[4] / image_count*100)))
                    LogWriter.write_to_logfile('k=10: {} of {} -> {}%'.format(values[5], image_count, int(values[5] / image_count*100)))
                    LogWriter.write_to_logfile('k=20: {} of {} -> {}%'.format(values[6], image_count, int(values[6] / image_count*100)))
                    LogWriter.write_to_logfile('k=30: {} of {} -> {}%'.format(values[7], image_count, int(values[7] / image_count*100)))
                    LogWriter.write_to_logfile('')
                    LogWriter.write_to_logfile('')

                    # calc results for test scenarios

                    # Best Perspective
                    if source=="cam0_C" and destination == "cam1_C":
                        test_scenarios["Best Perspective Matching"][0] = test_scenarios["Best Perspective Matching"][0] + image_count
                        test_scenarios["Best Perspective Matching"][1] = test_scenarios["Best Perspective Matching"][1] + values[1]
                        test_scenarios["Best Perspective Matching"][2] = test_scenarios["Best Perspective Matching"][2] + values[2]
                        test_scenarios["Best Perspective Matching"][3] = test_scenarios["Best Perspective Matching"][3] + values[3]
                        test_scenarios["Best Perspective Matching"][4] = test_scenarios["Best Perspective Matching"][4] + values[4]
                        test_scenarios["Best Perspective Matching"][5] = test_scenarios["Best Perspective Matching"][5] + values[5]
                        test_scenarios["Best Perspective Matching"][6] = test_scenarios["Best Perspective Matching"][6] + values[6]
                        test_scenarios["Best Perspective Matching"][7] = test_scenarios["Best Perspective Matching"][7] + values[7]

                    # Same Perspective
                    if source.split("_")[-1] == destination.split("_")[-1]:
                        test_scenarios["Same Perspective Matching"][0] = test_scenarios["Same Perspective Matching"][0] + image_count
                        test_scenarios["Same Perspective Matching"][1] = test_scenarios["Same Perspective Matching"][1] + values[1]
                        test_scenarios["Same Perspective Matching"][2] = test_scenarios["Same Perspective Matching"][2] + values[2]
                        test_scenarios["Same Perspective Matching"][3] = test_scenarios["Same Perspective Matching"][3] + values[3]
                        test_scenarios["Same Perspective Matching"][4] = test_scenarios["Same Perspective Matching"][4] + values[4]
                        test_scenarios["Same Perspective Matching"][5] = test_scenarios["Same Perspective Matching"][5] + values[5]
                        test_scenarios["Same Perspective Matching"][6] = test_scenarios["Same Perspective Matching"][6] + values[6]
                        test_scenarios["Same Perspective Matching"][7] = test_scenarios["Same Perspective Matching"][7] + values[7]

                    # Cross Matching
                    if source.split("_")[0] != destination.split("_")[0]\
                       and (source.split("_")[-1] != destination.split("_")[-1]):
                        test_scenarios["Cross Matching"][0] = test_scenarios["Cross Matching"][0] + image_count
                        test_scenarios["Cross Matching"][1] = test_scenarios["Cross Matching"][1] + values[1]
                        test_scenarios["Cross Matching"][2] = test_scenarios["Cross Matching"][2] + values[2]
                        test_scenarios["Cross Matching"][3] = test_scenarios["Cross Matching"][3] + values[3]
                        test_scenarios["Cross Matching"][4] = test_scenarios["Cross Matching"][4] + values[4]
                        test_scenarios["Cross Matching"][5] = test_scenarios["Cross Matching"][5] + values[5]
                        test_scenarios["Cross Matching"][6] = test_scenarios["Cross Matching"][6] + values[6]
                        test_scenarios["Cross Matching"][7] = test_scenarios["Cross Matching"][7] + values[7]

                    # Full Cross Matching
                    test_scenarios["Full Cross Matching"][0] = test_scenarios["Full Cross Matching"][0] + image_count
                    test_scenarios["Full Cross Matching"][1] = test_scenarios["Full Cross Matching"][1] + values[1]
                    test_scenarios["Full Cross Matching"][2] = test_scenarios["Full Cross Matching"][2] + values[2]
                    test_scenarios["Full Cross Matching"][3] = test_scenarios["Full Cross Matching"][3] + values[3]
                    test_scenarios["Full Cross Matching"][4] = test_scenarios["Full Cross Matching"][4] + values[4]
                    test_scenarios["Full Cross Matching"][5] = test_scenarios["Full Cross Matching"][5] + values[5]
                    test_scenarios["Full Cross Matching"][6] = test_scenarios["Full Cross Matching"][6] + values[6]
                    test_scenarios["Full Cross Matching"][7] = test_scenarios["Full Cross Matching"][7] + values[7]

                # track results for test scenatios
                for test_scenario in test_scenarios.keys():
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=1)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=2)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=3)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=5)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=10)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=20)
                    mlflow.log_metric(key="images_{}".format(test_scenario), value=test_scenarios[test_scenario][0], step=30)

                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][1]/test_scenarios[test_scenario][0]*100, step=1)
                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][2]/test_scenarios[test_scenario][0]*100, step=2)
                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][3]/test_scenarios[test_scenario][0]*100, step=3)
                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][4]/test_scenarios[test_scenario][0]*100, step=5)
                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][5]/test_scenarios[test_scenario][0]*100, step=10)
                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][6]/test_scenarios[test_scenario][0]*100, step=20)
                    mlflow.log_metric(key=test_scenario, value=test_scenarios[test_scenario][7]/test_scenarios[test_scenario][0]*100, step=30)

                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][1], step=1)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][2], step=2)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][3], step=3)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][4], step=5)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][5], step=10)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][6], step=20)
                    mlflow.log_metric(key="absolute_{}".format(test_scenario), value=test_scenarios[test_scenario][7], step=30)

                    LogWriter.write_to_logfile(test_scenario)
                    LogWriter.write_to_logfile('k=1: {} of {} -> {}%'.format(test_scenarios[test_scenario][1], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][1]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('k=2: {} of {} -> {}%'.format(test_scenarios[test_scenario][2], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][2]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('k=3: {} of {} -> {}%'.format(test_scenarios[test_scenario][3], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][3]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('k=5: {} of {} -> {}%'.format(test_scenarios[test_scenario][4], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][4]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('k=10: {} of {} -> {}%'.format(test_scenarios[test_scenario][5], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][5]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('k=20: {} of {} -> {}%'.format(test_scenarios[test_scenario][6], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][6]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('k=30: {} of {} -> {}%'.format(test_scenarios[test_scenario][7], test_scenarios[test_scenario][0], int(test_scenarios[test_scenario][7]/test_scenarios[test_scenario][0]*100)))
                    LogWriter.write_to_logfile('')
                    LogWriter.write_to_logfile('')

                # measure and track computation time
                end_time = time.perf_counter()
                elapsed_time = '{:.2f} s'.format(end_time - start_time)
                mlflow.log_param('computation time', elapsed_time)
                self.print_logged_info(run)

    def write_signatures(self):

        _detection_pipeline = detection_pipeline.DetectionPipeline(
            model_path='../tests/models/resnet_detection.pth',
            model_path_reid=os.path.join(
                os.path.join(save_path, self.backbone),
                'model/model.pth.tar-{}'.format(self.epochs)),
            vector_creator="PersonReId",
            distance_measurement="cosine",
            backbone=self.backbone
        )

        dataset = PalletDataset(
            root='../tests'
        )

        images = [image[0] for image in dataset.gallery]
        vectors = _detection_pipeline.vector_class.create_vector(images)

        logfile_folder = os.path.join(save_path, self.backbone)
        os.makedirs(logfile_folder, exist_ok=True)
        with open(os.path.join(logfile_folder, 'feature_vectors.csv'), 'w+') as file:
            for i in range(len(images)):
                split = images[i].split('_')
                pallet_id = int(split[-2]) - 1
                csv_string = '{0};{1};{2}'.format(images[i], pallet_id, list(vectors[i]))
                file.write(csv_string + '\n')

    def write_signatures_industry(self, reg_frame="2"):
        batch_size = 100

        _detection_pipeline = detection_pipeline.DetectionPipeline(
            model_path='../tests/models/resnet_detection.pth',
            model_path_reid=os.path.join(
                os.path.join(save_path, self.backbone),
                'model/model.pth.tar-{}'.format(self.epochs)),
            vector_creator="PersonReId",
            distance_measurement="cosine",
            backbone=self.backbone
        )

        dataset = PalletDataset(
            root='../tests',
            val=True
        )
        images = [image[0] for image in dataset.train]
        images_query = [image[0] for image in dataset.query]
        images_gallery = [image[0] for image in dataset.gallery]
        images.extend(images_query)
        images.extend(images_gallery)

        images_by_frame = {
            "1": [],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
        }

        for filename in images:
            split = filename.split("_")
            frame = split[-1][:-4]
            images_by_frame[str(frame)].append(filename)

        registration_set = 0
        reidentification_set = 0
        registration_images = []
        reidentification_images = []

        for key in images_by_frame.keys():
            if key == str(reg_frame) or "C" == str(reg_frame):
                registration_set += len(images_by_frame[key])
                registration_images.extend(images_by_frame[key])
            else:
                reidentification_set += len(images_by_frame[key])
                reidentification_images.extend(images_by_frame[key])


        logfile_folder = os.path.join(save_path, self.backbone)
        os.makedirs(logfile_folder, exist_ok=True)

        with open(os.path.join(logfile_folder, 'feature_vectors_test.csv'), 'w+') as file:
            img_count = len(reidentification_images)
            batches = math.ceil(img_count / batch_size)

            for batch in tqdm(range(batches)):
                images_names = [reidentification_images[i] for i in
                              range(batch * batch_size, min(batch * batch_size + batch_size, img_count))]
                pil_images = [np.array(Image.open(reidentification_images[i])) for i in
                              range(batch * batch_size, min(batch * batch_size + batch_size, img_count))]
                pallet_ids = [int(reidentification_images[i].split("_")[-2]) - 1 for i in
                              range(batch * batch_size, min(batch * batch_size + batch_size, img_count))]
                vectors = _detection_pipeline.vector_class.create_vector(pil_images)

                for i in range(len(images_names)):
                    csv_string = '{0};{1};{2}'.format(images_names[i], pallet_ids[i], list(vectors[i]))
                    file.write(csv_string + '\n')


    def set_model(self, model):
        self.extractor.model = model

    def create_vector(self, images=[]):
        features = self.extractor(images)
        return features.cpu().numpy()

    def print_logged_info(self, r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))
        LogWriter.write_to_logfile("run_id: {}".format(r.info.run_id))
        LogWriter.write_to_logfile("artifacts: {}".format(artifacts))
        LogWriter.write_to_logfile("params: {}".format(r.data.params))
        LogWriter.write_to_logfile("metrics: {}".format(r.data.metrics))
        LogWriter.write_to_logfile("tags: {}".format(tags))


if __name__ == '__main__':
    backbone = 'pcb_p4'
    torchreid.models.show_avai_models()
    reid = PersonReId(backbone=backbone, run_testcases=True)
    #reid.train()
    #reid.run_tests()
    reid.run_simple_tests_single_cam(reg_frame="2")
    # reid.write_signatures()
    # reid.write_signatures_industry("2")
