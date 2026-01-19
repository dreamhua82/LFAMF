# Robust light field angular super-resolution via multi-dimensional feature fusion and attention-guided refinement
This repository contains official pytorch implementation of Robust light field angular super-resolution via multi-dimensional feature fusion and attention-guided refinement

## Requirement
* Ubuntu 18.04
* Python 3.6
* Pyorch 1.7
* Matlab

## Dataset
Please first download light field datasets, and put them into corresponding folders. The real-world training data is available in [SIGGRAPH/ACM Trans. Graph.](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/).

## Training
* Run:
  ```python
  python train.py
## Test
* Run:
  ```python
  python test.py

```
## Acknowledgement
Our work and implementations are based on the following projects: <br> 
[LF-EASR](https://github.com/GaoshengLiu/LF-EASR)<br> 
[LFASR-geo](https://github.com/jingjin25/LFASR-geometry)<br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
We sincerely thank the authors for sharing their code and amazing research work!
