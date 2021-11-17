from torchvision import transforms

import numbers
import numpy as np
from PIL import ImageFilter

from torchvision.datasets import ImageFolder
from knockoff.datasets.caltech256 import Caltech256
from knockoff.datasets.cifarlike import CIFAR10, CIFAR100, SVHN, TinyImagesSubset
from knockoff.datasets.cubs200 import CUBS200
from knockoff.datasets.diabetic5 import Diabetic5
from knockoff.datasets.imagenet1k import ImageNet1k
from knockoff.datasets.imagenet_224 import imagenet_224
from knockoff.datasets.imagenet_128 import imagenet_128
from knockoff.datasets.imagenet_64 import imagenet_64
from knockoff.datasets.imagenet_32 import imagenet_32
from knockoff.datasets.imagenet_28 import imagenet_28


from knockoff.datasets.imagenet_28_b60k import imagenet_28_b60k
from knockoff.datasets.imagenet_32_b60k import imagenet_32_b60k
from knockoff.datasets.imagenet_64_b60k import imagenet_64_b60k
from knockoff.datasets.imagenet_128_b60k import imagenet_128_b60k
from knockoff.datasets.imagenet_224_b60k import imagenet_224_b60k
from knockoff.datasets.imagenet_331_b60k import imagenet_331_b60k

from knockoff.datasets.imagenet_32_b20k import imagenet_32_b20k

from knockoff.datasets.cifar100_28 import cifar100_28
from knockoff.datasets.cifar100_32 import cifar100_32
from knockoff.datasets.cifar100_64 import cifar100_64
from knockoff.datasets.cifar100_128 import cifar100_128
from knockoff.datasets.cifar100_224 import cifar100_224

from knockoff.datasets.indoor67_28 import indoor67_28
from knockoff.datasets.indoor67_32 import indoor67_32
from knockoff.datasets.indoor67_64 import indoor67_64
from knockoff.datasets.indoor67_128 import indoor67_128
from knockoff.datasets.indoor67_224 import indoor67_224

from knockoff.datasets.cubs200_28 import cubs200_28
from knockoff.datasets.cubs200_32 import cubs200_32
from knockoff.datasets.cubs200_64 import cubs200_64
from knockoff.datasets.cubs200_128 import cubs200_128
from knockoff.datasets.cubs200_224 import cubs200_224

from knockoff.datasets.caltech256_28 import caltech256_28
from knockoff.datasets.caltech256_32 import caltech256_32
from knockoff.datasets.caltech256_64 import caltech256_64
from knockoff.datasets.caltech256_128 import caltech256_128
from knockoff.datasets.caltech256_224 import caltech256_224


from knockoff.datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST
from knockoff.datasets.tinyimagenet200 import TinyImageNet200

from torchvision import datasets 

# Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/11
class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception("`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception("radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception("`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))


# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply
dataset_to_modelfamily = {
    # MNIST
    'MNIST': 'mnist',
    'KMNIST': 'mnist',
    'EMNIST': 'mnist',
    'EMNISTLetters': 'mnist',
    'FashionMNIST': 'mnist',

    # Cifar
    'cifar10': 'cifar',
    'cifar100': 'cifar',
    'SVHN': 'cifar',
    'TinyImageNet200': 'cifar',
    'TinyImagesSubset': 'cifar',

    # Imagenet
    'CUBS200': 'imagenet',
    'caltech256': 'imagenet',
    'indoor67': 'imagenet',
    'Diabetic5': 'imagenet',
    'Imagenet': 'imagenet',
    'imagenet_28': 'imagenet_28',
    'imagenet_32': 'imagenet_32',
    'imagenet_64': 'imagenet_64',
    'imagenet_128': 'imagenet_128',
    'imagenet_224': 'imagenet_224',
    
    #Imbalance
    'imagenet_28_im60k': 'imagenet_28',
    'imagenet_32_im60k': 'imagenet_32',
    'imagenet_64_im60k': 'imagenet_64',
    'imagenet_128_im60k': 'imagenet_128',
    'imagenet_224_im60k': 'imagenet_224',
    'imagenet_331_im60k': 'imagenet_331',

    'open_28_im60k': 'imagenet_28',
    'open_32_im60k': 'imagenet_32',
    'open_64_im60k': 'imagenet_64',
    'open_128_im60k': 'imagenet_128',
    'open_224_im60k': 'imagenet_224',
    'open_331_b60k': 'imagenet_331',
    
 
    #Balance 
    'imagenet_28_b90k': 'imagenet_28',
    'imagenet_32_b90k': 'imagenet_32',
    'imagenet_64_b90k': 'imagenet_64',
    'imagenet_128_b90k': 'imagenet_128',
    'imagenet_224_b90k': 'imagenet_224',
    'imagenet_331_b90k': 'imagenet_331',
     

    'imagenet_28_b60k': 'imagenet_28',
    'imagenet_32_b60k': 'imagenet_32',
    'imagenet_64_b60k': 'imagenet_64',
    'imagenet_128_b60k': 'imagenet_128',
    'imagenet_224_b60k': 'imagenet_224',
    'imagenet_331_b60k': 'imagenet_331',

    'imagenet_32_b20k': 'imagenet_32',
    
    'imagenet_28_b30k': 'imagenet_28',
    'imagenet_32_b30k': 'imagenet_32',
    'imagenet_64_b30k': 'imagenet_64',
    'imagenet_128_b30k': 'imagenet_128',
    'imagenet_224_b30k': 'imagenet_224',
    'imagenet_331_b30k': 'imagenet_331',

    'ImageFolder': 'imagenet',

    'cubs200_28': 'imagenet_28',
    'caltech256_28': 'imagenet_28',
    'indoor67_28': 'imagenet_28',
    'cifar10_28' : 'cifar_28',
    'cifar100_28' : 'cifar_28',
 
    
    'cubs200_32': 'imagenet_32',
    'caltech256_32': 'imagenet_32',
    'indoor67_32': 'imagenet_32',
    'cifar10_32' : 'cifar_32',
    'cifar100_32' : 'cifar_32',


    'cubs200_64': 'imagenet_64',
    'caltech256_64': 'imagenet_64',
    'indoor67_64': 'imagenet_64',
    'cifar10_64' : 'cifar_64',
    'cifar100_64' : 'cifar_64',


    'cubs200_128': 'imagenet_128',
    'caltech256_128': 'imagenet_128',
    'indoor67_128': 'imagenet_128',
    'cifar10_128' : 'cifar_128',
    'cifar100_128' : 'cifar_128',


    'cubs200_224': 'imagenet_224',
    'caltech256_224': 'imagenet_224',
    'indoor67_224': 'imagenet_224',
    'cifar10_224' : 'cifar_224',
    'cifar100_224' : 'cifar_224',
 
}

modelfamily_to_mean_std = {
    'mnist': {
        'mean': (0.1307,),
        'std': (0.3081,),
    },
    'cifar': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
    },
    'imagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
}

sir_model_dimension = {
    
    'wres28-1':32,
    'wres28-5':32,    
    'wres28-10':32,
    'res50_28':28,
    'res50_32':32,
    'res50_64':64,
    'res50_128':128,
    'res50_224':224,
    'res50_331':331, 
}

num_class={
    'cubs200_28': 200,
    'caltech256_28': 256,
    'indoor67_28': 67,
    'cifar10_28' : 10,
    'cifar100_28' : 100,
    
    'cubs200_32': 200,
    'caltech256_32': 256,
    'indoor67_32': 67,
    'cifar10_32' : 10,
    'cifar100_32' : 100,
 
    'cubs200_64': 200,
    'caltech256_64': 256,
    'indoor67_64': 67,
    'cifar10_64' : 10,
    'cifar100_64' : 100,
 
    'cubs200_128': 200,
    'caltech256_128': 256,
    'indoor67_128': 67,
    'cifar10_128' : 10,
    'cifar100_128' : 100,

    'cubs200_224': 200,
    'caltech256_224': 256,
    'indoor67_224':67,
    'cifar10_224' : 10,
    'cifar100_224' : 100,



}


# Transforms
modelfamily_to_transforms = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    },


    'cifar_28': {
        'train': transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
      	    transforms.Resize((28, 28)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
    },

    'cifar_32': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
      	    transforms.Resize((32, 32)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
    },


    'cifar_64': {
        'train': transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
      	    transforms.Resize((64, 64)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
    },


    'cifar_128': {
        'train': transforms.Compose([
            transforms.RandomCrop(128, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
      	    transforms.Resize((128, 128)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
    },


    'cifar_224': {
        'train': transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
      	    transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
    },


    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet_28': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet_32': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet_64': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet_128': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'imagenet_224': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },


}
