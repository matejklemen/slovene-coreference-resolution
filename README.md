# Slovene coreference resolution

Slovene coreference resolution project introduces four models for coreference resolution on coref149 and senticoref datasets:

- baseline model (linear regression with hand-crafted features),
- non-contextual neural model using word2vec embeddings,
- contextual neural model using ELMo embeddings,
- contextual neural model using BERT embeddings.

For more details, [see the report paper](https://github.com/matejklemen/slovene-coreference-resolution/blob/master/report/Coreference_resolution_approaches_for_Slovene_language.pdf).

## Run it in Google Colaboratory

Easiest way to run this project is opening and running the [prepared notebook in Google Colab](https://colab.research.google.com/github/matejklemen/slovene-coreference-resolution/blob/master/report/Slovene_coreference_resolution.ipynb).

## Project structure

- `report/` contains the pdf of our work.
- `src/` contains the source code of our work.
- `data/` is a placeholder for datasets (see _Getting datasets_ section below).

## Setup

Before doing anything, the dependencies need to be installed.  
```bash
$ pip install -r requirements.txt
```

NOTE: if you have problems with `torch` library, make sure you have python x64 installed. Also make use of 
[this](https://pytorch.org/get-started/locally/#start-locally) official command builder.

### Getting datasets

The project operates with the following datasets: 
- [SSJ500k](https://www.clarin.si/repository/xmlui/handle/11356/1210) (`-sl.TEI` version), 
- [coref149](https://www.clarin.si/repository/xmlui/handle/11356/1182)
- [sentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285) ([WebAnno TSV 3.2 File format docs](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20(GitHub)%20(master)/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#sect_webannotsv)).

Download and extract them into `data/` folder. After that, your data folder should look like this:
```
data/
+-- ssj500k-sl.TEI
    +-- ssj500k.back.xml
    +-- ssj500k-sl.body.xml
    +-- ssj500k-sl.xml
    +-- ...
+-- coref149
    +-- ssj4.15.tcf
    +-- ssj5.30.tcf
    +-- ... (list of .tcf files)
+-- senticoref1_0
    +-- 1.tsv
    +-- 2.tsv
    +-- ... (list of .tsv files)
```

Coref149 and SentiCoref are the main datasets we use. 

SSJ500k is used for additional metadata such as dependencies and POS tags, which are not provided by coref149 itself.

Since only a subset of SSJ500k is used, it can be trimmed to decrease its size and improve loading time. 
To do that, run `trim_ssj.py`:
```bash
$ python src/trim_ssj.py --coref149_dir=data/coref149 --ssj500k_path=data/ssj500k-sl.TEI/ssj500k-sl.body.xml --target_path=data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml
```

If `target_path` parameter is not provided, the above command would produce 
`data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml`.

If you want to use pretrained embeddings in non-contextual coreference model, make sure to download the Slovene
- word2vec vectors (`Word2Vec Continuous Skipgram`) from http://vectors.nlpl.eu/repository/ and/or
- fastText vectors (`bin`) from https://fasttext.cc/docs/en/crawl-vectors.html (not used in the paper but supported)

Put them into `data/` (either the `cc.sl.300.bin` file for fastText or the `model.txt` file for word2vec).

For the contextual coreference model, make sure to download the pretrained Slovene ELMo embeddings from 
https://www.clarin.si/repository/xmlui/handle/11356/1277. 
Extract the options file and the weight file into `data/slovenian-elmo`.


## Running the project

Before running anything, make sure to set `DATA_DIR` and `SSJ_PATH` parameters in `src/data.py` file (if the paths to 
where your datasets are stored differ in your setup).

Below are examples how to run each model.

`--dataset` parameter can be either `coref149` or `senticoref`.

### Baseline model

```bash
$ python baseline.py \
  --model_name="my_baseline_model" \
  --learning_rate="0.05" \
  --dataset="coref149" \
  --num_epochs="20" \
  --fixed_split
```

### Non-contextual model

```bash
$ python noncontextual_model.py \
    --model_name="my_noncontextual_model" \
    --fc_hidden_size="512" \
    --dropout="0.0" \
    --learning_rate="0.001" \
    --dataset="coref149" \
    --embedding_size="100" \
    --use_pretrained_embs="word2vec" \
    --freeze_pretrained \
    --fixed_split
```

### Contextual model (ELMo)

```bash
$ python contextual_model_elmo.py \
    --model_name="my_elmo_model" \
    --dataset="coref149" \
    --fc_hidden_size="64" \
    --dropout="0.4" \
    --learning_rate="0.005" \
    --num_epochs="20" \
    --freeze_pretrained \
    --fixed_split
```

### Contextual model (BERT)

```bash
$ python contextual_model_bert.py \
    --model_name="my_bert_model" \
    --dataset="coref149" \
    --fc_hidden_size="64" \
    --dropout="0.4" \
    --learning_rate="0.001" \
    --num_epochs="20" \
    --freeze_pretrained \
    --fixed_split
```

# License

