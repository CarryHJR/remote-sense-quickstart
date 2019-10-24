## change-detection

### tutorial posts

https://medium.com/@carryhjr/remote-sense-change-detection-quick-start-65d00a89622b

### tutorial code

change-detection-quick-start.ipynb

### datasets

Onera Satellite Change Detection Dataset - https://rcdaudt.github.io/oscd/ - 14 pairs - 13 spectral - multi resolution(10,20,60) - 2018

air change dataset - http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html - 13 paris - rgb - 2009

dataset in a paper - https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf

damage change dataset  - https://github.com/gistairc/ABCDdataset

### current method

The Eearly-Fusion architecture concatenated the two patches before passing them through the net-work, treating them as different color channels.

The Siamese architecture processed both images separately at first by iden-tical branches of the network with shared structure and pa- rameters, merging the two branches only after the convolu-tional layers of the network.

The tranditional method is also popular, like "iterative slow feature analysis"

