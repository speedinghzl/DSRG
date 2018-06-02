# Working in Progress

We are updating our code. Please do not clone this repo yet.

# Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing (CVPR2018)


## Installing dependencies

* Python packages:
```bash
      $ pip install -r python-dependencies.txt
```
* **caffe**: installation instructions are available at `http://caffe.berkeleyvision.org/installation.html`.
   Note, you need to compile **caffe** with python wrapper and support for python layers.

* Fully connected CRF wrapper (requires the **Eigen3** package).
```bash
      $ pip install CRF/
```

## Training the DSRG model

* Go into the training directory: 

```bash
      $ cd training
      $ mkdir localization_cues
```

* Download the initial [VGG16](https://drive.google.com/open?id=1nq49w4os6BZ1JcrM4xqZKZh1wR3-32wi) model pretrained on Imagenet: 

* Download CAM [seed](https://drive.google.com/open?id=1cHyhjul9srPlwcl4xqrYR9MwzhFGwKXU) and put it in *training/localization_cues* folder. We use [CAM](http://cnnlocalization.csail.mit.edu/) for localizing the foreground seed classes and utilize the saliency detection technology [DRFI](http://supermoe.cs.umass.edu/~hzjiang/drfi/) for localizing background seed. We provide the python interface to DRFI [here](https://github.com/speedinghzl/drfi_cpp) for convenience.

```bash
      $ cd training/experiment/seed_mc
      $ mkdir models
```
* Set *root_folder* parameter in **train-s.prototxt, train-f.prototxt** and *PASCAL_DIR*  in **run-s.sh** to the directory with **PASCAL VOC 2012** images

* Run:

```bash
      $ bash run-s.sh
```
   The trained model will be created in `models`
   
## Acknowledgment
This code is heavily borrowed from [SEC](https://github.com/kolesman/SEC).
