## Introdunction

This code is based on Knockoffnet code, and input dimension experiment is additionally implemented. 
The modified code is displayed in the directory tree below.


```bash
├── data #Dataset here
│   ├── caltech256
│   ├── cifar100
│   ├── cubs200
│   ├── imagenet60000
│   └── indoor
├── knockoff
│   ├── adversary
│   │   ├── adaptive.py
│   │   ├── __init__.py
│   │   ├── jacobian.py
│   │   ├── train_at.py #Implement Activethief uncertainty attack with pytorch
│   │   ├── train.py #Add image resizing code
│   │   └── transfer.py
│   ├── config.py
│   ├── datasets #Dataset for each dimension
│   │   ├── caltech256_128.py
│   │   ├── caltech256_224.py
│   │   ├── caltech256_28.py
│   │   ├── caltech256_32.py
│   │   ├── caltech256_64.py
│   │   ├── caltech256.py
│   │   ├── cifar100_128.py
│   │   ├── cifar100_224.py
│   │   ├── cifar100_28.py
│   │   ├── cifar100_32.py
│   │   ├── cifar100_64.py
│   │   ├── cifarlike.py
│   │   ├── cubs200_128.py
│   │   ├── cubs200_224.py
│   │   ├── cubs200_28.py
│   │   ├── cubs200_32.py
│   │   ├── cubs200_64.py
│   │   ├── cubs200.py
│   │   ├── diabetic5.py
│   │   ├── flowers.py
│   │   ├── imagenet_128_b60k.py
│   │   ├── imagenet_128.py
│   │   ├── imagenet1k.py
│   │   ├── imagenet_224_b60k.py
│   │   ├── imagenet_224.py
│   │   ├── imagenet_28_b60k.py
│   │   ├── imagenet_28.py
│   │   ├── imagenet_32_b20k.py
│   │   ├── imagenet_32_b60k.py
│   │   ├── imagenet_32.py
│   │   ├── imagenet_331_b60k.py
│   │   ├── imagenet_64_b60k.py
│   │   ├── imagenet_64.py
│   │   ├── indoor67_128.py
│   │   ├── indoor67_224.py
│   │   ├── indoor67_28.py
│   │   ├── indoor67_32.py
│   │   ├── indoor67_64.py
│   │   ├── indoor67.py
│   │   ├── __init__.py
│   │   ├── mnistlike.py
│   │   └── tinyimagenet200.py
│   ├── fidelity.py # Compute fidelity 
│   ├── __init__.py
│   ├── utils 
│   │   ├── folder.py
│   │   ├── __init__.py
│   │   ├── model.py # Add each dimension model code (res50_32, res50_64, res50_128, res50_224)
│   │   ├── resnet_128.py
│   │   ├── resnet_32.py
│   │   ├── resnet_64.py
│   │   ├── resnet.py
│   │   ├── transforms.py
│   │   ├── type_checks.py
│   │   ├── utils.py
│   │   └── wresnet.py
│   └── victim
│       ├── blackbox.py
│       ├── __init__.py
│       └── train.py
└── models #Model saved here
    ├── adversary
    ├── pretrained
    └── victim
```

## Train Victim Model
Bash script for training the victim model (e.g. res50 [224] , indoor67 )

    #!/bin/bash                                                                                                                                                                                  
    
    {
    #knockoff                                                                                                                                                                                    
    victim_model='res50_224'
    victimset='indoor67_224'
    victim_model_dir="models/victim/$victimset/$victim_model"
    out_dir="models/victim/$victimset/$victim_model"
    batch_size=32
    gpu='0'
    epoch=200
    lr=0.1
    pretrained='imagenet'

    python -u knockoff/victim/train.py --dataset $victimset --model_arch $victim_model --out_path $victim_model_dir -d $gpu -b $batch_size -e $epoch --lr $lr --pretrained $pretrained
    
    }



## Attack Command  
This tutorial written with reference to [](https://github.com/tribhuvanesh/knockoffnets#transfer-set-construction)Knockoff training 



### Transfer Set Construction


    #!/bin/bash                                                                                                                                                                                  
    
    {
    #knockoff                                                                                                                                                                                    
    policy='random'
    victim_model='res50_224'
    victimset='indoor67_224'
    victim_model_dir="models/victim/$victimset/$victim_model"
    queryset='imagenet_224_b60k'
    out_dir="models/adversary/$victimset/$victim_model/$queryset"
    batch_size=8
    gpu='1'
    budget='60000'
    echo '************victim ' $victim_model $victimset
    echo $victim_model $victim_model_dir $victimset $out_dir $queryset $budget $batch_size
    
    python -u knockoff/adversary/transfer.py --policy $policy --victim_model_dir $victim_model_dir --victim_model $victim_model --out_dir $out_dir --budget $budget --queryset $queryset --tests\
    et $victimset --batch_size $batch_size -d $gpu
    
    }

### Knockoffnets attack

    #!/bin/bash                                                                                                                                                                                  
    {
    #knockoff                                                                                                                                                                                    
    queryset='imagenet_224_b60k'
    victim_model='res50_224'
    testdataset='indoor67_224'
    victim_model_dir="models/victim/$testdataset/$victim_model"
    model_save_dir="models/adversary/$testdataset/$victim_model/$queryset"
    surrogate_model='res50_224'
    origdataset='indoor67_224'
    gpu='0'
    budget='60000'
    epochs=200
    lr=0.01
    wd=0
    momentum=0.5
    pretrained='imagenet'
    echo '************victim ' $victim
    echo '************surrogate ' $sur
    echo $victim_model $victim_model_dir $victimset $out_dir $queryset $budget $batch_size
    
    python -u knockoff/adversary/train.py --model_dir $model_save_dir --model_arch $surrogate_model --testdataset $testdataset --origdataset $origdataset --budgets $budget -d $gpu  --log-inter\
    val 100 --epochs $epochs --lr $lr --wd $wd --momentum $momentum --victim_model $victim_model --victim_model_dir $victim_model_dir --pretrained $pretrained
    
    
    }



### Activetheif attack

    #!/bin/bash                                                                                                                                                                                  
    {
    #knockoff                                                                                                                                                                                    
    queryset='imagenet_224_b90k'
    victim_model='res50_64'
    testdataset='indoor67_64'
    victim_model_dir="models/victim/$testdataset/$victim_model"
    model_save_dir="models/adversary/$testdataset/$victim_model/$queryset"
    surrogate_model='res50_224'
    origdataset='indoor67_224'
    gpu='0'
    budget='20000'
    epochs=200 # the number of epoch per round
    lr=0.01
    wd=0
    momentum=0.5
    pretrained='imagenet'
    echo '************victim ' $victim
    echo '************surrogate ' $sur
    echo $victim_model $victim_model_dir $victimset $out_dir $queryset $budget $batch_size
    
    python -u knockoff/adversary/train_at.py --model_dir $model_save_dir --model_arch $surrogate_model --testdataset $testdataset --origdataset $origdataset --budgets $budget -d $gpu  --log-in\
    terval 100 --epochs $epochs --lr $lr --wd $wd --momentum $momentum --victim_model $victim_model --victim_model_dir $victim_model_dir --pretrained $pretrained
    
    
    }






## Reference code

```
@inproceedings{orekondy19knockoff,
    TITLE = {Knockoff Nets: Stealing Functionality of Black-Box Models},
    AUTHOR = {Orekondy, Tribhuvanesh and Schiele, Bernt and Fritz, Mario},
    YEAR = {2019},
    BOOKTITLE = {CVPR},
}

@inproceedings{pal2020activethief,
  title={Activethief: Model extraction using active learning and unannotated public data},
  author={Pal, Soham and Gupta, Yash and Shukla, Aditya and Kanade, Aditya and Shevade, Shirish and Ganapathy, Vinod},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={01},
  pages={865--872},
  year={2020}
}
```








