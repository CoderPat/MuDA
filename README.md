# Multidimensional Discourse-Aware (MuDA) Benchmark

**TODO**: Write introduction here.

## Installation

The tagger relies on Pytorch (`<1.10`) to run models. If you want to run these models, first install Pytorch. You can find instructions for your system [here](https://pytorch.org/get-started/locally/).

For example, to install PyTorch on a Linux system with CUDA support in a conda environment, run:

```bash
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Then, to install the rest of the dependencies, run:

```bash
pip install -r requirements.txt
```

## Example Usage

```bash
python muda/tagger.py \
    --src example_data/dev.en \
    --tgt example_data/dev.de \
    --docids example_data/dev.docids 
    --tgt-lang de
```