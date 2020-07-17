# Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN

## Bibtex

    @InProceedings{Lan_2019_CVPR,
        author = {Lan, Shiyi and Yu, Ruichi and Yu, Gang and Davis, Larry S.},
        title = {Modeling Local Geometric Structure of 3D Point Clouds Using Geo-CNN},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }

## Installation and Usage

We re-implemented the Geo-CNN following [Frustum PointNets](https://github.com/charlesq34/frustum-pointnets).


Follow the instruction of installing Frustum PointNets and thus compile Geo-Conv operator located at models/tf\_ops/geoconv.

Use scripts/command\_train\_geocnn\_v1.sh and command\_test\_geocnn\_v1.sh to train/test Geo-CNN.

## TODO

* Combine GeoCNN and PointNet++
* GeoCNN on other 3D datasets (ModelNet40, ScanNet)

## Others

* Well-trained [parameters](https://drive.google.com/open?id=15hq1E61li7fAgTt_0AIW7mc_1HdFNwAJ)
* This implementation is slightly different from the original version on a private deep learning architure.
