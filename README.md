# F-SV: A novel approach for filtering various structural variants via Multi-Feature Fusion
## Introduction
Chromosomal structural variation refers to significant structural changes that occur on a chromosome. Compared to single nucleotide variations, chromosomal structural variations have a higher probability of causing diseases, particularly chromosomal disorders and their exacerbation. Accurate identification of chromosomal structural variations can greatly enhance the effectiveness of preventing disease occurrence through prenatal diagnosis.<br>
However, the study of structural variations using second-generation sequencing technologies is unable to detect large-scale structural variations and certain types of structural variations. The existing third-generation sequencing technologies have emerged in recent years as a new generation of sequencing technologies. Third-generation sequencing technologies, such as PacBio and Nanopore, offer higher throughput and longer read lengths, enabling the discovery of structural variations that cannot be detected using second-generation sequencing data. This presents a new opportunity for studying chromosomal structural variations.<br>
This paper proposes a structural variation identification model, F-SV, based on multi-feature fusion to address the limitations of existing general-purpose models in obtaining comprehensive sequence alignment feature signals, low efficiency and accuracy in identifying structural variations, and the inability to handle certain alignments. In this method, certain non-universal and ineffective signals are eliminated, and different signal encoding methods are proposed for various structural variation signals to fully extract their features. The stability and robustness of the method and model are demonstrated through 5-fold cross-validation. Compared to DeepSVFilter, the proposed method achieves an improvement in accuracy of over 3.36\%, enabling more accurate and efficient identification of chromosomal structural variations.

# Requirements

F-SV is tested to work under:

* Python 3.6

# Quick start
## Add channels
```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

## Environment by using anaconda and pip
```shell
conda create -n F-SV python=3.6 -y
conda activate F-SV
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -y
conda install pytorch-lightning=1.5.10 -c conda-forge -y
pip install ray[tune] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pysam==0.15.4
conda install hyperopt -y
conda install redis -y
conda install scikit-learn -y
conda install matplotlib -y
conda install samtools -c bioconda -y
conda install pudb -y

```