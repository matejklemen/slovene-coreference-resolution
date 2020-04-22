""" Remove documents from SSJ500k which are not used in coref149 - to reduce file reading time. """

import logging
import argparse
import os
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description="Remove unneeded documents from ssj500k")
parser.add_argument("--coref149_dir", type=str, required=True, help="Directory path of coref149 corpus")
parser.add_argument("--ssj500k_path", type=str, required=True, help="FULL path to the SSJ500k XML file")
parser.add_argument("--target_path", type=str, help="Path where reduced SSJ500k dataset will be stored")

if __name__ == "__main__":
    args = parser.parse_args()
    corpus_dir = args.coref149_dir

    logging.info(f"**Reading valid document IDs from {corpus_dir}**")
    coref149_ids = [f[:-4] for f in os.listdir(corpus_dir)
                    if os.path.isfile(os.path.join(corpus_dir, f)) and f.endswith(".tcf")]

    logging.info(f"**Reading SSJ500k corpus from {args.ssj500k_path}**")
    with open(args.ssj500k_path, encoding="utf8") as ssj:
        content = ssj.readlines()
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")

    logging.info(f"**Removing redundant documents**")
    for curr_doc in soup.findAll("p"):
        if curr_doc["xml:id"] not in coref149_ids:
            curr_doc.decompose()

    assert len(soup.findAll("p")) == 149

    target_path = args.target_path
    if not target_path:
        # append ".reduced" to filename, i.e. ssj500k-sl.body.xml -> ssj500k-sl.body.reduced.xml
        path, ext = os.path.splitext(args.ssj500k_path)
        target_path = path + ".reduced" + ext

    logging.info(f"**Saving reduced SSJ500k dataset to {target_path}**")
    with open(target_path, "w", encoding="utf8") as f_target:
        f_target.write(str(soup))
