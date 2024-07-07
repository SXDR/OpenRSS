# Open-Vocabulary RGB-Thermal Semantic Segmentation  



This is the official repository of our paper:

**Abstract:** RGB-Thermal (RGB-T) semantic segmentation is an important research branch of multi-modal image segmentation. The current RGB-T semantic segmentation methods generally have two unsolved and typical shortcomings. First, they do not have the open-vocabulary recognition ability, which significantly limits their application scenarios. Second, when fusing RGB and thermal images, they often need to design complex fusion network structures, which usually results in low network training efficiency. We present OpenRSS, the **Open**-vocabulary **R**GB-T **S**emantic **S**egmentation method, to solve these two disadvantages. To our knowledge, OpenRSS is the first RGB-T semantic segmentation method with open-vocabulary segmentation capability. OpenRSS modifies the basic segmentation model SAM for RGB-T semantic segmentation by adding the proposed thermal information prompt module and dynamic low-rank adaptation strategy to SAM. These designs effectively fuse the RGB and thermal information, but with much fewer trainable parameters than other methods. OpenRSS achieves the open-vocabulary capability by jointly utilizing the vision-language model CLIP and the modified SAM. Through extensive experiments, OpenRSS demonstrates its effective open-vocabulary semantic segmentation ability on RGB-T images. It outperforms other state-of-the-art RGB open-vocabulary semantic segmentation methods on multiple RGB-T semantic segmentation benchmarks: +12.1% mIoU on the MFNet dataset, +18.4% mIoU on the MCubeS dataset, and +21.4% mIoU on the Freiburg Thermal dataset. 

![openrss](C:\Users\ZGQ\Downloads\openrss.png)

# Code and checkpoints

Coming soon!