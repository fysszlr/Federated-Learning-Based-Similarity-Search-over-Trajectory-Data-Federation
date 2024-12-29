## Federated-Learning-Based-Similarity-Search-over-Trajectory-Data-Federation

### Requirments

* CentOS Linux release 7.9.2009 (Core)
* `pip install -r reqirments.txt`
* dataset: the Beijing Taxi Dataset

### Dataset

* For privacy reasons, we are unable to provide the Beijing taxi dataset. If you wish to use the Porto dataset, you can download it from [Porto Taxi Trajectory Data](https://www.kaggle.com/datasets/crailtap/taxi-trajectory).

### Data processing

#### Porto
* use `bash scripts/data_processing.sh`

#### Beijing
1. use utils/beijing2traj.py to convert the original trajectory points of the Beijing dataset into trajectories. 
2. save the file obtained in the first step to data/beijing.csv. 
3. use utils/preprocessing_beijing.py to process the data.

### Run
> If you are using beijing dataset, you may modify all 'dataset' field in scripts to 'beijing'.
#### ourmethod

* distort
  * use `bash scripts/run_fed-trajCl_distort.sh`
* downsampling
  * use `bash scripts/run_fed-trajCl_downsampling.sh`
* hit_ratio
  * use `bash scripts/run_fed-trajCl_simi.sh`

#### fedavg

* distort
  * use `bash scripts/run_fedavg_distort.sh`
* downsampling
  * use `bash scripts/run_fedavg_downsampling.sh`
* hit_ratio
  * use `bash scripts/run_fedavg_simi.sh`
