# LeaFTL 

LeaFTL is a learning-based flash translation layer (FTL), which learns the address mapping and tolerates dynamic data access patterns via linear regression at runtime to reduce the memory footprint of the address mapping.

## 1. System Setup

The following packages and benchmarks are necessary to install before running the experiments. It is recommended to allocate a dedicated server for these experiments as they are both cpu and memory intensive.

```shell
cd LeaFTL
pushd .
# Download traces
cd wiscsee/leaftl_scripts
gdown 1Lw0DgZWwaeuSckLnKqoBAcM64rlGFWZi
unzip traces.zip

popd
# Set Env Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/wiscsee
```



## 2. Simulation-based Experiments

### 2.1 Comparison v.s. DFTL and SFTL

```shell
cd LeaFTL/wiscsee/leaftl_scripts
# Run warmup
./warmup main_batch
# Run batch of experiments
./batch main_batch
# Run plot scripts
./plot_all main_batch
```

### 2.2 Expected Results

Here we will decribe the expected output logs and figures.