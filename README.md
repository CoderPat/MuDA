# Multilingual Discourse-Aware (MuDA) Benchmark

The Multilingual Discourse-Aware (MuDA) Benchmark is a comprehensive suite of taggers and evaluators aimed at advancing the field of context-aware Machine Translation (MT). 

Traditional translation quality metrics output uninterpertable scores, and fail to accuratly measure performance on context-aware discourse phenomena. MuDA takes a different direction, relying on neural-based syntatical and morphalogical analysers to measure performance of translation models on specific words and discourse phenomena.

The MuDA taggers currently support 14 language pairs (see [this directory](CoderPat/MuDA/muda/langs)) but easily supports adding new languages.

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

To tag an existing dataset, and extract the tags for later use, run the following command. 

```bash
python muda/main.py \
    --src /path/to/src \
    --tgt /path/to/tgt \
    --docids /path/to/docids \
    --dump-tags /tmp/maia_ende.tags \
    --tgt-lang "$lang" \
```

To evaluate models on particular dataset (reporting per-tag metrics such as precision & recall), run

```bash
python muda/main.py \
    --src /path/to/src \
    --tgt /path/to/tgt \
    --docids /path/to/docids \
    --hyps /path/to/hyps.m1 /path/to/hyps.m2 \
    --tgt-lang "$lang"
```

Note that MuDA relies on an `docids` file, containing the same number of lines as the `src/tgt` files and where each line contains a *document id* to which the source/target in the line belong to.
