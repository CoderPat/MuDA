# Multidimensional Discourse-Aware (MuDA) Benchmark

**TODO**: Write introduction here.

## Installation

**TODO**: Write installation instructions here.

## Example Usage

```bash
python $repo/scripts/stanza_tokenize.py $test_src $test_src_tok --lang $src_lang
python $repo/scripts/stanza_tokenize.py $test_tgt $test_tgt_tok --lang $tgt_lang

python $repo/scripts/format_align.py \
    --source-file $test_src_tok \
    --target-file $test_tgt_tok \
    --output formatted_output

awesome-align \
    --output_file=alignments.out \
    --model_name_or_path=$awesome_model \
    --data_file=formatted_output \
    --extraction ‘softmax’ \
    --batch_size 32  \
    --cache_dir $awesome_cachedir

python $repo/scripts/tagger.py \
    --src-tok-file $test_src_tok \
    --tgt-tok-file $test_tgt_tok \
    --src-detok-file $test_src \
    --tgt-detok-file $test_tgt \
    --docids-file $test_docids \
    --alignments-file alignments.out \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --source-context-size 1000 \
    --target-context-size 1000 \
    --output $tagged_tgt
```