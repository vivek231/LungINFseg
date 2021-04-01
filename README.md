# LungINFseg
[![](https://img.shields.io/badge/python-3.6%2B-green.svg)]()
LungINFseg: Segmenting COVID-19 Infected Regions in Lung CT Images
Based on Receptive-Field-Aware Deep Learning Framework 

## Prerequisites
```
+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
+ pytorch 4.0/4.1
+ torchvision
```
## Getting started
```
- Clone this repo 
- Install requirements
```


**+ Train the model:**

    python train.py 
    
**+ Test the model:**

    python test.py


:point_down: Screenshot:

<p align="center">
  <img src="/static/screenshot.png" height="480px" alt="">
</p>

## Citation:
If you use the code in your work, please use the following citation:
```
@article{kumar2021lunginfseg,
  title={Lunginfseg: Segmenting covid-19 infected regions in lung ct images based on a receptive-field-aware deep learning framework},
  author={Kumar Singh, Vivek and Abdel-Nasser, Mohamed and Pandey, Nidhi and Puig, Domenec},
  journal={Diagnostics},
  volume={11},
  number={2},
  pages={158},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
