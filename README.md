# Sliceformer 


This repo is the official implementation of "[Sliceformer: Make Multi-head Attention as Simple as Sorting in Discriminative Tasks](https://arxiv.org/abs/2310.17683)".

We provide the Sliceformer code for various tasks, including the LRA benchmark, image classification, text classification, and molecular property prediction.

The [LRA](https://github.com/DaShenZi721/sliceformer/tree/master/LRA) directory contains the code for [the Long-Range Arena benchmark](https://github.com/google-research/long-range-arena).

The [sliceformer](https://github.com/DaShenZi721/sliceformer/tree/master/sliceformer) directory covers tasks related to image classification, which includes datasets such as CIFAR-10, CIFAR-100, MNIST, and the Dogs vs. Cats dataset. It also includes text classification tasks with the IMDB dataset.

The [Graphormer](https://github.com/DaShenZi721/sliceformer/tree/master/Graphormer) directory is dedicated to tasks involving molecular property prediction, specifically on the PCQM4M-LSC dataset.

For more detailed information, please refer to the README files in each of these three directories.

## Citation

If you find out work useful, please cite our paper at:

```
@misc{yuan2023sliceformer,
      title={Sliceformer: Make Multi-head Attention as Simple as Sorting in Discriminative Tasks}, 
      author={Shen Yuan and Hongteng Xu},
      year={2023},
      eprint={2310.17683},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```