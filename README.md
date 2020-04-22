# slovene-coreference-resolution

TODO: A short introduction/description of what this repository is about.

# Project structure

- `report/` contains the pdf of our work.
- `src/` contains the source code of our work.
- `data/` is a placeholder for datasets (see _Getting datasets_ section below).

# Setup

Before doing anything, the dependencies need to be installed.  
```bash
$ pip install -r requirements.txt
```

NOTE: if you have problems with `torch` library, make sure you have python x64 installed. Also make use of 
[this](https://pytorch.org/get-started/locally/#start-locally) official command builder.

## Getting datasets

The project operates with the following datasets: 
- [SSJ500k](https://www.clarin.si/repository/xmlui/handle/11356/1210) (`-sl.TEI` version), 
- [coref149](https://www.clarin.si/repository/xmlui/handle/11356/1182).

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
```

Coref149 is the main dataset we use, while the other one is used for additional metadata such as dependencies 
and POS tags.

Since only a subset of SSJ500k is used, it can be trimmed to decrease its size and improve loading time. 
To do that, run `trim_ssj.py`:
```bash
$ python src/trim_ssj.py --coref149_dir=data/coref149 --ssj500k_path=data/ssj500k-sl.TEI/ssj500k-sl.body.xml
```

If `target_path` parameter is not provided, the above command would produce 
`data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml`.


# Running the project

Before running anything, make sure to set `DATA_DIR` and `SSJ_PATH` parameters in `src/data.py` file (if the paths to 
where your datasets are stored differ in your setup).

To run the baseline model (with hand-crafted features), run `baseline.py`.
```bash
$ python src/baseline.py

```

TBD additional steps as more things get added.

# Dev notes

Each document is stored inside a `Document` object (see `data.py`). It stores various information about the document 
such as the actual text, tokens, (ground truth) mentions, (ground truth) coreference clusters, additional metadata 
from SSJ500k etc..  

More specifically, it contains the following properties:  
- `doc_id` contains the document ID
- `id_to_tok` contains mapping from token IDs to raw tokens;
- `sents` contains compressed sentences, which are stored as lists of token IDs;
- `mentions` contains mapping from mention IDs to `Mention` objects (contains metadata for mentions);
- `clusters` contains list of mention clusters. Each cluster is a list of mention IDs;
- `mapped_clusters` contains mapping from a mention to its antecedent (or dummy antecedent). It is a convenience 
property that represents the same thing as `clusters`, but in a different way (as a *coreference chain*);
- `ssj_doc` contains a BeautifulSoup object (XML in a nicer form) with additional metadata, extracted from SSJ500k.

To (hopefully) make things more clear, here's a code sample.
```python
>>> from data import corpus_read
# make sure to provide path to the reduced ssj500k dataset as 2nd argument!
>>> all_docs = corpus_read("<path to coref149 dir>", "<path to ssj50k dir>/ssj500k-reduced.xml")
>>> doc = all_docs[0]
>>> doc.doc_id
'ssj211.1399'
>>> doc.mentions
{'rc_0': <data.Mention at 0x7f22b3c80780>,
 'rc_1': <data.Mention at 0x7f22b3c80748>, ... }
# Obtain mention 'rc_1' in a human-readable form 
>>> doc.mentions["rc_1"].raw
['Strokovnjak', 'za', 'poslovanje', 'z', 'nepremičninami', 'iz', 'Maribora']
# Obtain token IDs for the same mention
>>> doc.mentions["rc_1"].token_ids
['ssj211.1399.5073.t1',
 'ssj211.1399.5073.t2',
 'ssj211.1399.5073.t3', ...]
```
