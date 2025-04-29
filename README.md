# Hierarchical Transformer for Image Captioning

## Introduction

This project presents a hierarchical Transformer architecture for image captioning, designed to generate accurate and descriptive captions. Our model combines a multi-scale hierarchical Transformer encoder, leveraging an enhanced ViT-CNN hybrid approach, with an enhanced Transformer decoder. The hierarchical encoder processes visual information at multiple scales, capturing both fine-grained details and global context. The decoder then generates captions by attending to these rich, multi-scale visual features. Experiments on standard image captioning benchmarks demonstrate our model's ability to produce detailed and contextually relevant captions. 

## Requirements

The code was tested using Python 3.8.12. Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

To run the analysis notebook (`src/experiment.ipynb`), you might also need to download the Stanza English model:
```python
import stanza
stanza.download("en")
```

## Run

### Create Dataset

Download the MS COCO 2017 dataset:
- Train images: [http://images.cocodataset.org/zips/train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
- Validation images: [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- Annotations: [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

Run the `src/create_dataset.py` script to process images, tokenize captions, create the vocabulary, and split the data. Adjust arguments like `dataset_dir` and `output_dir` as needed.

```bash
python src/create_dataset.py --dataset_dir <path_to_coco> --output_dir <output_path> [OTHER_ARGUMENTS]
```

This script saves processed images (`*.hdf5`), tokenized captions (`*.json`), caption lengths (`*.json`), and the vocabulary (`vocab.pth`) to the specified `output_dir`.

### Train the Model

Run the `src/run_train.py` script to train the model. Set the `dataset_dir` to the `output_dir` from the previous step and provide a configuration file path.

```bash
python src/run_train.py --dataset_dir <output_path> --config_path src/config.json [OTHER_ARGUMENTS]
```


### Test the Model (Inference)

Run the `src/inference_vit_cnn.py` script to generate captions for the test split images using beam search. Set the `dataset_dir` and provide the path to the trained model checkpoint.

```bash
python src/inference_vit_cnn.py --dataset_dir <output_path> --checkpoint_name <checkpoint_file.pth> --save_dir <results_dir> [OTHER_ARGUMENTS]
```

The script outputs a pandas DataFrame containing generated captions, ground truth, attention weights, and evaluation metrics (BLEU, GLEU, METEOR) to the `save_dir`.

## References

<a id="1">[1]</a> Liu, W., Chen, S., Guo, L., Zhu, X., & Liu, J. (2021). CPTR: Full transformer network for image captioning. arXiv preprint [arXiv:2101.10804](https://arxiv.org/abs/2101.10804).

<a id="2">[2]</a> Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014, September). Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755). Springer, Cham.
