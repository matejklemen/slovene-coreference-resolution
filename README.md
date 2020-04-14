# nlp-coreference-resolution

# Project structure
`report/` contains the pdf of our work (WIP).

# Setup
Before doing anything, the dependencies need to be installed.  
```bash
$ pip install -r requirements.txt
```

The project operates with 2 datasets: [SSJ500k](https://www.clarin.si/repository/xmlui/handle/11356/1210) 
and [coref149](https://www.clarin.si/repository/xmlui/handle/11356/1182), so make sure to download them (download the 
`sl TEI` version of ssj500k).
Coref149 is the main dataset we use, while the other one is used for additional metadata such as dependencies 
and POS tags.

Since only a subset of SSJ500k is used, it can be trimmed to decrease its size and improve loading time. 
To do that, run `trim_ssj.py`.
```bash
# target_path is optional - if not provided, `ssj500k_path` will be overriden
$ python src/trim_ssj --coref149_dir="..." \
    --ssj500k_path=".../ssj500k-sl.body.xml" \
    --target_path="..."
```

TBD additional steps as more things get added.

# Dev notes

Each document is stored inside a `Document` object (see `data.py`). It stores various information about the document 
such as the actual text, tokens, (ground truth) mentions, (ground truth) coreference clusters, 
additional metadata from SSJ500k etc..  

More specifically, it contains the following properties:  
- `id_to_tok` contains mapping from token IDs to raw tokens;
- `sents` contains compressed sentences, which are stored as lists of token IDs;
- `mentions` contains mapping from mention IDs to start and end index of mention tokens. 
For example, `{"rc_3": [13, 15]}` represents a mention `rc_3` that is made up of tokens at indices 13 and 14;
- `clusters` contains list of mention clusters. Each cluster is a list of mention IDs;
- `mapped_clusters` contains mapping from a mention to its antecedent. It is a convenience property that represents the 
same thing as `clusters`, but in a different way (as a *coreference chain*);
- `ssj_doc` contains a BeautifulSoup object (XML in a nicer form) with additional metadata, extracted from SSJ500k.

To (hopefully) make things more clear, here's a code sample.
```python
>>> from data import corpus_read
# make sure to provide path to the reduced ssj500k dataset as 2nd argument!
>>> all_docs = corpus_read("<path to coref149 dir>", "<path to ssj50k dir>/ssj500k-reduced.xml")
>>> doc = all_docs[0]
>>> doc.name
'.../ssj211.1399.tcf'
>>> doc.mentions
{'rc_0': [170, 171],
 'rc_1': [14, 21], ...}
# Obtain mention 'rc_1' in a human-readable form 
>>> flattened_doc = []
>>> for sent in doc.raw_sentences():
        flattened_doc.extend(sent)
>>> start, end = doc.mentions['rc_1']
>>> flattened_doc[start: end]
['Strokovnjak', 'za', 'poslovanje', 'z', 'nepremičninami', 'iz', 'Maribora']
```