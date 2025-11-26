# Code for FCL-SSMR
> Federated Continual Learning over Class-Overlapping Data Stream via Selective Synthetic Memory Replay

## Requairments
The needed libraries are in requirements.txt.

## Datasets
- **MNIST**  
  - **Description**: Handwritten digit recognition dataset.  
  - **Version**: Original (70,000 images, 60,000 training + 10,000 testing).  
  - **Download**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 

- **CIFAR-10**  
  - **Description**: 60,000 32x32 color images in 10 classes.  
  - **Version**: v1.0 (50,000 training + 10,000 testing).  
  - **Download**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 

- **CIFAR-100**  
  - **Description**: 60,000 32x32 color images in 100 classes.  
  - **Version**: v1.0 (50,000 training + 10,000 testing).  
  - **Download**: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)  

- **Tiny-ImageNet**
  - **Description**: 110,000 64x64 color images in 200 classes; 500 training + 50 validation per class.
  - **Version**: v1.0 (100,000 training + 10,000 validation).
  - **Download**: [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

- **Fashion-MNIST**
  - **Description**: 70,000 28x28 grayscale images of 10 fashion categories.
  - **Version**: v1.0 (60,000 training + 10,000 testing).
  - **Download**: [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz)

- **LEAF**
  - **Description**: A benchmark for federated learning featuring multiple datasets (FEMNIST Sentiment140, etc.) with natural user partitioning.
  - **Version**: v1.0 (varies by sub-dataset).
  - **Download**: [LEAF Framework](https://github.com/TalwalkarLab/leaf/archive/master.zip)

- **Leeds Sports Pose**
  - **Description**:A benchmark dataset for human pose estimation, containing 2,000 sports images with detailed pose annotations.
  - **Version**: v1.0 (1,000 training + 1,000 testing).
  - **Download**: [Leeds Sports Pose Dataset](https://opendatalab.org.cn/OpenDataLab/LSP
  )

- **SHIPS**
  - **Description**: Maritime vessel images for ship classification and detection tasks.
  - **Version**: v1.0 (15,000+ images across 10 ship categories).
  - **Download**: [SHIPS Dataset](https://www.kaggle.com/datasets/arpitjain007/ship-dataset)


## Experiments
To run on MNIST, excuate:

    python main.py --dataset=mnist --method=ours --task=3 --beta=0 --g_rounds=1000 --kd=5 --overlap_rate=0.25

To run on CIFAR-10, excuate:

    python main.py --dataset=cifar10 --method=ours --task=3 --beta=0 --g_rounds=2000 --kd=5 --overlap_rate=0.25

To run on CIFAR-100, excuate:

    python main.py --dataset=cifar100 --method=ours --task=5 --beta=0 --g_rounds=2200 --kd=10 --overlap_rate=0.4 --m_rate=0.6

To run on Tiny-ImageNet, excuate:

    python main.py --dataset=tiny_imagenet --method=ours --task=10 --beta=0 --g_rounds=2000 --kd=10 --overlap_rate=0.4 --m_rate=0.4

To run on Fashion-MNIST, excuate:

    python main.py --dataset=fmnist --method=ours --task=3 --beta=0 --g_rounds=1000 --kd=5 --overlap_rate=0.25

To run on LEAF, excuate:

    python main.py --dataset=leaf --method=ours --task=3 --beta=0 --g_rounds=1800 --kd=5 --overlap_rate=0.3

To run on Leeds Sports Pose, excuate:

    python main.py --dataset=lsp --method=ours --task=3 --beta=0 --g_rounds=1500 --kd=5 --overlap_rate=0.25

To run on SHIPS, excuate:

    python main.py --dataset=ships --method=ours --task=3 --beta=0 --g_rounds=1500 --kd=5 --overlap_rate=0.25