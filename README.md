# Slovene coreference resolution
Code to run the experiments, presented in the paper **Neural coreference resolution for Slovene language**.

In the paper, we analyze coreference resolution systems on two Slovene datasets, _coref149_ (small) and 
_SentiCoref 1.0_ (bigger). We also provide a detailed description of the latter as this is the first 
analysis on it.  
We explore the following models:
- a simple linear baseline with hand-crafted features from existing literature,
- non-contextual word embeddings (word2vec and fastText),
- contextual ELMo embeddings,
- contextual BERT embeddings (trilingual and multilingual BERT).

**Limitations:** Note that we consider only the coreference resolution task, i.e. we assume the mentions are detected in advance and provided.

For more details, [see the journal paper](https://doi.org/10.2298/CSIS201120060K),
to appear in Computer Science and Information Systems (ComSIS).

## Setup

Before doing anything, the dependencies need to be installed.  
```bash
$ pip install -r requirements.txt
```

**Note**: you might have problems running the ELMo contextual models on Windows since [allennlp](https://github.com/allenai/allennlp#installation) is not officialy supported on Windows, as noted in their README.

### Getting datasets

The project operates with the following datasets: 
- [SSJ500k](https://www.clarin.si/repository/xmlui/handle/11356/1210) (`-sl.TEI` version), 
- [coref149](https://www.clarin.si/repository/xmlui/handle/11356/1182),
- [sentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285).

Download and extract them into `data/` folder. 
```shell
$ cd data/
$ # COREF149
$ wget https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1182/coref149_v1.0.zip
$ unzip -q coref149_v1.0.zip -d coref149
$ rm coref149_v1.0.zip
$ # ssj500k
$ wget https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1210/ssj500k-sl.TEI.zip
$ unzip -q ssj500k-sl.TEI.zip
$ rm ssj500k-sl.TEI.zip
$ # SentiCoref 1.0
$ wget https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1285/SentiCoref_1.0.zip
$ unzip -q SentiCoref_1.0.zip -d senticoref1_0
$ rm SentiCoref_1.0.zip
```

After that, your data folder should look like this:
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

_coref149_ and _SentiCoref 1.0_ are the main datasets we use.
_ssj500k_ is used for additional metadata such as POS tags, which are not provided by coref149 itself.

Since only a subset of SSJ500k is used, it can be trimmed to decrease its size and improve loading time. 
To do that, run `trim_ssj.py`:
```bash
$ # assuming current directory is 'data/'
$ cd ..
$ python src/trim_ssj.py --coref149_dir=data/coref149 --ssj500k_path=data/ssj500k-sl.TEI/ssj500k-sl.body.xml --target_path=data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml
```

If `target_path` parameter is not provided, the above command will produce 
`data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml`.

### Getting embeddings
If you want to use pretrained embeddings in non-contextual coreference model, make sure to download the Slovene
- word2vec vectors (`Word2Vec Continuous Skipgram`) from http://vectors.nlpl.eu/repository/ and/or
- fastText vectors (`bin`) from https://fasttext.cc/docs/en/crawl-vectors.html

Put them into `data/` (either the `cc.sl.300.bin` file for fastText or the `model.txt` file for word2vec).
```shell
$ cd data/
$ # word2vec
$ mkdir word2vec
$ wget http://vectors.nlpl.eu/repository/20/67.zip
$ unzip -q 67.zip -d word2vec/
$ rm 67.zip
$ # fastText
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.bin.gz
$ gunzip cc.sl.300.bin.gz
$ rm cc.sl.300.bin.gz
```

**IMPORTANT**: to train a model based on fastText, you need to run `reduce_fasttext.py`, which reduces the 
size of downloaded embeddings by only keeping word vectors to be used in datasets in advance.


For the contextual coreference model, make sure to download the pretrained Slovene ELMo embeddings from 
https://www.clarin.si/repository/xmlui/handle/11356/1277. 
Extract the options file and the weight file into `data/slovenian-elmo`.
```shell
$ # assuming current directory is 'data/'
$ wget https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1277/slovenian-elmo.tar.gz
$ tar -xvzf slovenian-elmo.tar.gz
$ mv slovenian slovenian-elmo
$ rm slovenian-elmo.tar.gz
```

The BERT contextual embeddings will be downloaded automatically on the first run.


## Running the project

Below are examples how to run each model. First, move into the `src/` directory:
```shell
$ # assuming current directory is 'data/'
$ cd ../src
```

Parameters and their default values can be previewed with the `--help` flag. 
`--dataset` can be either `coref149` or `senticoref`.

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
    --embedding_size=100 \
    --use_pretrained_embs="word2vec" \
    --embedding_path="../data/word2vec/model.txt" \
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



