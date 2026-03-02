## Efficient Traffic Flow Forecasting via Sequence-Aware Node Routing and Cluster-Based Spatial Aggregation
 
<p align="center">
  <img src="model_arch.png" width="400"  alt="SRCA">
</p>

#### Performance on Traffic Forecasting Benchmarks   

<p align="center">

  <img src="result_datasets.png" width="800"  alt="Performance on Traffic Datesets"> 

</p>

#### Computational Cost between Different Models

<p align="center">

  <img src="computation_datasets.png" width="800"  alt="Computational Cost on Traffic Datesets"> 

</p>

#### Required Packages

```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

#### Training Commands

```bash
cd model/
sh train_all.sh
```

`<dataset>`:
- METRLA
- PEMSBAY
- PEMS04
- PEMS07
- PEMS08
- PEMS07L
- PEMS07M
