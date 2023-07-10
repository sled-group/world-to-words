# Pre-training Instructions

## Data Preparation

> Note: Our pre-training dataset are inherited from [MDETR](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md) paper, you can also refer to their [pre-training instructions](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md) to prepare the dataset and potentially train the model with other datasets.

1. Download the original Flickr30k image dataset from [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) webpage and update the flickr_img_path to the folder containing the images.
2. Download the original Flickr30k entities annotations from [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities) and update the flickr_dataset_path to the folder with annotations.
3. Download MDETR's pre-processed annotations from [this link](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and update the flickr_ann_path to this folder with pre-processed annotations.

## Pre-training

We trained our model with 8 Nvidia A40 GPUs, each with 48GB of VRAM. The pre-training process took 3 days to finish with a step number of 150k.

The pre-training process is logged with [WandB](https://wandb.ai/site).

### W2-Bert

The `pretrain_flickr.sh` script contains the exact command we used to pre-train our model.

```bash
bash scripts/pretrain/pretrain_w2bert.sh
```

Note:
- For analysis purpose, we also saved a series of checkpoints during the pre-training process. If you don't want to save the checkpoints, you can remove the `--save_for_aoa` flag from the script.
