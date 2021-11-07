# Vitrual Try-On *Flask*

![20200118004320.png](https://raw.githubusercontent.com/GrayXu/Online-Storage/master/img/20200118004320.png)![20200118004342.png](https://raw.githubusercontent.com/GrayXu/Online-Storage/master/img/20200118004342.png)![20200118004359.png](https://raw.githubusercontent.com/GrayXu/Online-Storage/master/img/20200118004359.png)

A multi-stage virtual try-on deep neural networks based on [JPP-Net and CP-VTON]

Feature:
- Fast **Human_Image + Cloth_Image = Gen_Image**, put on a specified upper clothes for specified people.
- Web and Backend response service on *Flask*

# How to use

## Run complete server codes

1. Download 3 [pretrained models](#checkpoints) to `/checkpoints`.
2. install [dependency packages](#Dependency)
3. Start Flask service `python main.py`


~~*For now, GPU environment is essential. If you want to run it on CPU environment, delete all `.cuda()` calls in CPVTON.py and Networks.py.*~~   
CPU Mode: `Model(.., use_cuda = False)`

# Network Architecture

This multi-stage network consists of 3 parts:
- **JPP-Net**: Extract human features, make pose estimation & human parsing
- **Geometric Matching Module**: Input human features and clothes images, and twist clothes based on learned thin-plate-spline algorithm.
- **Try-on Module**: Input human feature and twisted clothes images, and generate try-on images.


![20200117182844.png](https://raw.githubusercontent.com/GrayXu/Online-Storage/master/img/20200117182844.png)

# File

Fname | Usage  
-|-  
main.py | Flask service  
get.py | clients post predict requests to Flask server  
Model.py | Virtual Try-on Net
CPVTON.py | CPVTON model (GMM+TOM)
networks.py | CPVTON's basic network unit
JPPNet.py | JPP-Net model's init & predict
static/* | static resource
data/\*, example/\* | example test images
template/\* | Flask's html file based on Jinjia
checkpoints/\* | checkpoints dir

# Checkpoints

~~*coming soon..*~~  
download link on [*Google Drive*](https://drive.google.com/file/d/125UtOS4T4RBji8lXtm9WEwD1KcHG4F1g/view?usp=sharing) and [*Baidu Pan*](https://pan.baidu.com/s/1e8tKEz7hpHAxqV6B5_hOIw)


# TODO List

- [x] Optimize model  
- [x] Web try-on service  
- [x] Basic documentation and comments  
- [x] Client post documentation  
- [x] Faster models download support  
- [x] CPU support

# References

Model designs are based on JPP-Net & CPVTON, and their open-source repo on Github.  

[**VITON**: An Image-based Virtual Try-on Network](https://arxiv.org/abs/1711.08447v1),Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, Larry S. Davis. **CVPR 2018**

[(**CP-VTON**) Toward Characteristic-Preserving Image-based Virtual Try-On Networks](https://arxiv.org/abs/1807.07688), Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, Meng Yang. **ECCV 2018**

[(**JPP-Net**) Look into Person: Joint Body Parsing & Pose Estimation Network and A New Benchmark](https://arxiv.org/abs/1804.01984), Xiaodan Liang, Ke Gong, Xiaohui Shen, Liang Lin, **T-PAMI 2018**.

----

*Powered By **Imba**, [JD Digits](https://www.jddglobal.com/)*
