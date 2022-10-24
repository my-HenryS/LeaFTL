# LeaFTL 

LeaFTL is a learning-based flash translation layer (FTL), which learns the address mapping and tolerates dynamic data access patterns via linear regression at runtime to reduce the memory footprint of the address mapping.

## 1. System Setup

The following packages and benchmarks are necessary to install before running the experiments. It is recommended to allocate a dedicated server for these experiments as they are both cpu and memory intensive.

```shell
git clone https://github.com/my-HenryS/LeaFTL.git
cd LeaFTL
pushd .
# Download traces
cd wiscsee/leaftl_scripts
gdown 1Lw0DgZWwaeuSckLnKqoBAcM64rlGFWZi
unzip traces.zip
popd

# Set Env Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/wiscsee

# Install Pypy to speedup the experiment running
# You can run WiscSim 2-3× faster with Pypy. You can install Pypy2 with this guide: https://doc.pypy.org/en/latest/install.html.
wget https://downloads.python.org/pypy/pypy2.7-v7.3.9-linux64.tar.bz2
tar xf pypy2.7-v7.3.9-linux64.tar.bz2
export PATH=$PATH:$(pwd)/pypy2.7-v7.3.9-linux64/bin

# Install Python packages
cd wiscsee/leaftl_scripts
./setup.sh
```



## 2. Simulation-based Experiments

### 2.1 Memory Reduction Comparison with Baseline FTLs

```shell
cd LeaFTL/wiscsee/leaftl_scripts
# Run batch of experiments
./batch memory_batch
# Run plot scripts
./plot_all memory_batch
```

#### 2.1.1 Expected Results

Here we will decribe the expected output logs and figures.

### 2.2 Performance Improvement Comparison with Baseline FTLs

```shell
cd LeaFTL/wiscsee/leaftl_scripts
# Run batch of experiments
./warmup performance_batch
# Run batch of experiments
./batch performance_batch
# Run plot scripts
./plot_all performance_batch
```

#### 2.2.1 Expected Results

Here we will decribe the expected output logs and figures.

### 2.3 Sensitivity Analsysis with Different Gamma

```shell
cd LeaFTL/wiscsee/leaftl_scripts
# Run batch of experiments
./warmup sensitivity_batch
# Run batch of experiments
./batch sensitivity_batch
# Run plot scripts
./plot_all sensitivity_batch
```

#### 2.3.1 Expected Results

Here we will decribe the expected output logs and figures.