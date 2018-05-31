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
```

* Download the initial [VGG16](http:www.baidu.com) model pretrained on Imagenet:

* Download CAM [seed](http:www.baidu.com) and put it in *training/localization_cues* folder

```bash
      $ cd training/experiment/seed_mc
```
* Set *root_folder* parameter in **train-s.prototxt, train-f.prototxt** and *PASCAL_DIR*  in **run-s.sh** to the directory with **PASCAL VOC 2012** images

* Run:

```bash
      $ bash run-s.sh
```
   The trained model will be created in `models`
