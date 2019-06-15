# Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN

## Bibtex

    @article{DBLP:journals/corr/abs-1811-07782,
      author    = {Shiyi Lan and
                  Ruichi Yu and
                  Gang Yu and
                  Larry S. Davis},
      title     = {Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN},
      journal   = {CoRR},
      volume    = {abs/1811.07782},
      year      = {2018},
      url       = {http://arxiv.org/abs/1811.07782},
      archivePrefix = {arXiv},
      eprint    = {1811.07782},
      timestamp = {Mon, 26 Nov 2018 12:52:45 +0100},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-07782},
      bibsource = {dblp computer science bibliography, https://dblp.org}
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
