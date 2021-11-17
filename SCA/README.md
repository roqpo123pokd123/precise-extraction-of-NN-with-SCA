This code is implemented to experiment with SCA.

## Prerequisites



### 1. Environment
 - Ubuntu 20.04
 - gcc 9.3.0
 - OpenBLAS 0.3.18 (https://github.com/xianyi/OpenBLAS/tree/v0.3.18)
 - libtorch v1.10.0 (https://github.com/pytorch/pytorch/tree/v1.10.0)

### 2. Set up OpenBLAS (GEMM library)

Compile and install to /opt/OpenBLAS.

```bash
$ git clone https://github.com/xianyi/OpenBLAS.git
$ cd ./OpenBLAS
$ make OPENBLAS_NUM_THREADS=1
$ make install PREFIX=/opt/OpenBLAS
```

### 3. Set up libtorch

Compile with OpenBLAS support as GEMM library.

```bash
$ git clone https://github.com/pytorch/pytorch.git
```

Set environment variable to use OpenBLAS in libtorch.

```bash
$ export USE_CUDA=0
$ export USE_CUDNN=0
$ export USE_MKLDNN=0
$ export USE_MKLDNN=0
$ export BLAS="OpenBLAS"
```

install pytorch

```bash
$ cd ./pytorch
$ python setup.py install
```

## Build experiment code

Download the experiment code.

```
$ git clone https://github.com/blingcho/SCA_MEA.git
```

Set environment variables for OpenBLAS and libtorch.

```
$ export BLAS=OpenBLAS
$ export OpenBLAS_HOME=/opt/OpenBLAS
$ export LD_LIBRARY_PATH=$OpenBLAS_HOME:$LD_LIBRARY_PATH
```

build projects

```bash
$ cd SCA/
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/pytorch/torch .. # your pytorch path
$ make
```



## Running cache-timing attack on OpenBLAS



### 1. get target addresses

When the batch size is small, OpenBLAS performs matrix multiplication using the sgemm itcopy, sgemm oncopy, and sgemm kernel functions.
An attacker can locate the address in /opt/OpenBLAS/libopenblas.so.0 and use mmap to obtain the virtual address.
Alternatively, for testing purposes, you can obtain addresses of actual target addresses from analyzing process address map with ASLR disabled.


Put target addresses in line 227~229 of SCA/sca.cpp in order.

```c
Line 226 : // insert target address
Line 227 : insert_target((void*)0x7fffxxxxxxxx); //itcopy 0
Line 228 : insert_target((void*)0x7fffxxxxxxxx); //oncopy 1
Line 229 : insert_target((void*)0x7fffxxxxxxxx); //kernel 2
```

### 2. set up conv layer

In our experiments, we employ each of the four models (variations of resnet depend on ID).  
Choose the conv layer with which you want to experiment. 

```c
// SCA/sca.cpp
Line 57 : //conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 7).stride(2).padding(3))); //224
Line 58 : //conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 3).stride(2).padding(1))); //128
Line 59 : //conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 4).stride(1).padding(1))); //64
Line 60 : conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 3).stride(1).padding(1))); //32
```

Make sure the input data for your choosed conv layer is correct.

```c
// SCA/sca.cpp
Line 195 : //at::Tensor inputTensor = torch::ones({1, 3,224,224});
Line 196 : //at::Tensor inputTensor = torch::ones({1, 3,128,128});
Line 197 : //at::Tensor inputTensor = torch::ones({1, 3,64,64});
Line 198 : at::Tensor inputTensor = torch::ones({1, 3,32,32});
```

### 3. running code

Run the experimental code.

```bash
$ ./build/sca
[2] Round 0
[1] Round 0
[2] start monitoring...
[1] start victim inference
[1] victim ends
[2] attacker ends...

[2] 0 round trace
itcopy 1389 72
oncopy 1404 60
kernel 1413 64
oncopy 1423 60
kernel 1424 66
oncopy 1426 58
kernel 1430 88
oncopy 1432 58
kernel 1433 64
oncopy 1441 58
kernel 1442 64
oncopy 1450 60
kernel 1451 62
itcopy 1456 70
kernel 1476 64
itcopy 1506 70
kernel 1520 82
itcopy 1549 70
kernel 1560 62
```

## Reference

```
@inproceedings{yan2020cache,
	title={Cache telepathy: Leveraging shared resource attacks to learn $\{$DNN$\}$ architectures},
    	author={Yan, Mengjia and Fletcher, Christopher W and Torrellas, Josep},
      	booktitle={29th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 20)},
        pages={2003--2020},
	year={2020}
	}

@inproceedings{yarom2014flush+,
	title={FLUSH+ RELOAD: A high resolution, low noise, L3 cache side-channel attack},
	author={Yarom, Yuval and Falkner, Katrina},
	booktitle={23rd $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 14)},
	pages={719--732},
	year={2014}
	}

@misc{malith jayaweera_19AD,
	     title={Hacking DNN Architectures in Cloud Environments},
	     url={https://malithjayaweera.com/2019/12/hacking-dnn-architectures-in-cloud-environments/},
	     author={Malith Jayaweera},
	     year={19AD},
	     month={Dec}}
```








