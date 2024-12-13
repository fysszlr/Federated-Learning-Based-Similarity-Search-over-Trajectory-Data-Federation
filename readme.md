## Federated-Learning-Based-Similarity-Search-over-Trajectory-Data-Federation

### Requirments

* CentOS Linux release 7.9.2009 (Core)
* `pip install -r reqirments.txt`
* dataset: the Beijing Taxi Dataset

### Data processing

* use `bash scripts/data_processing.sh`

### Run

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

