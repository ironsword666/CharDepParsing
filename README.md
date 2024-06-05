# Character-Level Chinese Dependency Parsing via Modeling Latent Intra-Word Structure

The source code for our 2024 ACL-findings paper: Character-Level Chinese Dependency Parsing via Modeling Latent Intra-Word Structure

## Requirements

**Python Env**

See `requirements.txt` and `environment.yaml` for the python environment.

**GPU**

All experiments can be run on a single 1080Ti GPU within 6 hours.

## Data

TODO

<!-- ### Stanford Dependencies

### Penn2Malt -->

## How to Run

### Pipeline Model

**Train**

```bash
# word segmentation
bash scripts/train_crf_cws_bert.sh
# word-level dependency parsing
bash scripts/train_crf_word_dep_bert.sh
```

**Evaluate**

```bash
bash scripts/eval_pipeline_bert.sh
```

### Joint Model (Character-level Dependency Parsing)

#### Train

**Latent**

```bash
bash scripts/train_latent_char_dep_bert.sh
```

**Latent-c2f**

```bash
bash scripts/train_latent_c2f_char_dep_bert.sh
```

**Leftward or Rightward**

```bash
# use `--orientation` to specify leftward or rightward
bash scripts/train_explicit_char_dep_bert.sh
```

#### Evaluate

Please refer to the `scripts/eval_latent_c2f_char_dep_bert.sh` for more details.


#### Evaluate with Gold Segmentation

Please refer to the `scripts/eval_joint_w_gold_seg_bert.sh` for more details.










