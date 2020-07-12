## change-detection

### code
https://github.com/CarryHJR/remote-sense-quickstart/blob/master/change-detection/change-detection-quick-start.ipynb
### tutorial
https://github.com/CarryHJR/remote-sense-quickstart/blob/master/change-detection/README.md
### datasets

Onera Satellite Change Detection Dataset - https://rcdaudt.github.io/oscd/ - 14 pairs - 13 spectral - multi resolution(10,20,60) - 2018

air change dataset - http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html - 13 paris - rgb - 2009

dataset in a paper - https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf

damage change dataset  - https://github.com/gistairc/ABCDdataset

### current method

The Eearly-Fusion architecture concatenated the two patches before passing them through the net-work, treating them as different color channels.

The Siamese architecture processed both images separately at first by iden-tical branches of the network with shared structure and pa-rameters, merging the two branches only after the convolu-tional layers of the network.

The tranditional method is also popular, like "iterative slow feature analysis"


## scene classification
### tutorial 
https://github.com/CarryHJR/remote-sense-quickstart/blob/master/scene-classification/REMDME.md

### code

https://github.com/CarryHJR/remote-sense-quickstart/blob/master/scene-classification/scene-classification-quickstart.ipynb

### dataset

1. UC Merced Land-Use Data Set
   contains 21 scene classes and 100 samples of size 256x256 in each class.
   http://weegee.vision.ucmerced.edu/datasets/landuse.html
2. WHU-RS19 Data Set
   has 19 different scene classes and 50 samples of size 600x600 in each class.
   http://captain.whu.edu.cn/repository.html

3. AID
   has 30 different scene classes and about 200 to 400 samples of size 600x600 in each class.
   https://captain-whu.github.io/AID/

4. NWPU-RESISC45 

   This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class
   http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

5. PatternNet 

   38 classes and each class has 800 images of size 256×256 pixels.

   https://drive.google.com/file/d/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K/view?usp=sharing

6. RSSCN7
   contains 7 scene classes and 400 samples of size 400x400 in each class.
   https://sites.google.com/site/qinzoucn/documents

### current method

Personlly, although so some papers are proposed every year, the best methods are raw deep netural network like SENet 154, EfficientNet, to name a few.

## scene classification attention visualization
### code
https://github.com/CarryHJR/remote-sense-quickstart/blob/master/view-attention/AID-view-attention.ipynb

### tutorial
https://github.com/CarryHJR/remote-sense-quickstart/blob/master/view-attention/README.md
