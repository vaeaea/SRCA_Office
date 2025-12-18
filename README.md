# SRCA_Office
Efficient Traffic Flow Forecasting via Sequence-Aware Node Routing and Cluster-Based Spatial Aggregation

Upload full code as soon as research is accepted
 
<p align="center">
  <img src="model_arch.png" width="300"  alt="SRCA">
</p>

#### Performance on Traffic Forecasting Benchmarks   

<p align="center">

  <img src="result_flow.png" width="500"  alt="Performance on Traffic Flow Dateset"> 

</p>

<p align="center"> 

  <img src="result_speed.png" width="500"  alt="Performance on Traffic Speed Dateset">

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
