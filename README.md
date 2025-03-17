# Character-Level Chinese Dependency Parsing via Modeling Latent Intra-Word Structure

The source code for our 2024 ACL-findings paper: Character-Level Chinese Dependency Parsing via Modeling Latent Intra-Word Structure

## Requirements

**Python Env**

See `requirements.txt` and `environment.yaml` for the python environment.

**GPU**

Each experiment can be run on a single 1080Ti GPU within 6 hours.

## Data

like:
```
1	法	_	NR	_	_	3	VMOD	_	_
2	正	_	AD	_	_	3	VMOD	_	_
3	研究	_	VV	_	_	0	ROOT	_	_
4	从	_	P	_	_	6	VMOD	_	_
5	波黑	_	NR	_	_	4	PMOD	_	_
6	撤军	_	VV	_	_	7	NMOD	_	_
7	计划	_	NN	_	_	3	VMOD	_	_

1	新华社	_	NR	_	_	8	DEP	_	_
2	巴黎	_	NR	_	_	8	DEP	_	_
3	９月	_	NT	_	_	8	DEP	_	_
4	１日	_	NT	_	_	8	DEP	_	_
5	电	_	NN	_	_	8	DEP	_	_
6	（	_	PU	_	_	8	P	_	_
7	记者	_	NN	_	_	8	DEP	_	_
8	张有浩	_	NR	_	_	0	ROOT	_	_
9	）	_	PU	_	_	8	P	_	_
```

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










