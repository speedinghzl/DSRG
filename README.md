# Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing (CVPR2018)
By [Zilong Huang](http://speedinghzl.github.io), [Xinggang Wang](http://mclab.eic.hust.edu.cn/~xwang/index.htm), [Jiasi Wang](https://github.com/JiasiWang), [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu) and [Jingdong Wang](https://jingdongwang2017.github.io/).

## Introduction
![Overview of DSRG](http://p9p8n5on3.bkt.clouddn.com/DSRG.PNG)
Overview of the proposed approach. The Deep Seeded Region Growing module takes the seed cues and segmentation map as input produces latent pixel-wise supervision which is more accurate and more complete than seed cues. Our method iterates between reﬁning pixel-wise supervision and optimizing the parameters of a segmentation network.


### License

DSRG is released under the MIT License (refer to the LICENSE file for details).

### Citing DSRG

If you find DSRG useful in your research, please consider citing:

    @inproceedings{huang2018dsrg,
        title={Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing},
        author={Huang, Zilong and Wang, Xinggang and Wang, Jiasi and Liu, Wenyu and Wang, Jingdong},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        pages={7014--7023},
        year={2018}
    }
    
## Installing dependencies

* Python packages:
```bash
      $ pip install -r python-dependencies.txt
```
* **caffe (deeplabv2 version)**: deeplabv2 caffe installation instructions are available at `https://bitbucket.org/aquariusjay/deeplab-public-ver2`. Note, you need to compile **caffe** with python wrapper and support for python layers. Then add the caffe python path into [training/tools/findcaffe.py](https://github.com/speedinghzl/DSRG/blob/master/training/tools/findcaffe.py#L21).

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

* Download the initial [VGG16](https://drive.google.com/open?id=1nq49w4os6BZ1JcrM4xqZKZh1wR3-32wi) model pretrained on Imagenet and put it in *training/* folder.

* Download CAM [seed](https://drive.google.com/open?id=1cHyhjul9srPlwcl4xqrYR9MwzhFGwKXU) and put it in *training/localization_cues* folder. We use [CAM](http://cnnlocalization.csail.mit.edu/) for localizing the foreground seed classes and utilize the saliency detection technology [DRFI](http://supermoe.cs.umass.edu/~hzjiang/drfi/) for localizing background seed. We provide the python interface to DRFI [here](https://github.com/speedinghzl/drfi_cpp) for convenience if you want to generate the seed by yourself.

```bash
      $ cd training/experiment/seed_mc
      $ mkdir models
```
* Set *root_folder* parameter in **train-s.prototxt, train-f.prototxt** and *PASCAL_DIR*  in **run-s.sh** to the directory with **PASCAL VOC 2012** images

* Run:

```bash
      $ bash run.sh
```
   The trained model will be created in `models`
   
   
## Acknowledgment
This code is heavily borrowed from [SEC](https://github.com/kolesman/SEC).
