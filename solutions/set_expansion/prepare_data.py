# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

"""
Script that prepares the input corpus for np2vec training: it runs NP extractor on the corpus and
marks extracted NP's.
"""

import logging
import sys
import json
import os
import datetime
import spacy
from configargparse import ArgumentParser
from nlp_architect.utils.io import check_size
from nlp_architect.utils.text_preprocess import spacy_normalizer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

np2id = {}
id2group = {}
id2rep = {}
np2count = {}

if __name__ == '__main__':
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--corpus',
        default='train.txt',
        type=str,
        action=check_size(min_size=1),
        help='path to the input corpus. By default, it is a subset of English Wikipedia dump.')
    arg_parser.add_argument(
        '--marked_corpus',
        default='marked_train.txt',
        type=str,
        action=check_size(min_size=1),
        help='path to the marked corpus corpus.')
    arg_parser.add_argument(
        '--mark_char',
        default='_',
        type=str,
        action=check_size(1, 2),
        help='special character that marks NP\'s in the corpus (word separator and NP suffix).')

    args = arg_parser.parse_args()

    corpus_file = open(args.corpus, 'r', encoding='utf8')
    marked_corpus_file = open(args.marked_corpus, 'w', encoding='utf8')

    # NP extractor using spacy
    logger.info('loading spacy')
    nlp = spacy.load('en_core_web_sm', disable=['textcat', 'ner'])
    logger.info('spacy loaded')

    num_lines = sum(1 for line in corpus_file)
    corpus_file.seek(0)
    logger.info('%i lines in corpus', num_lines)
    i = 0

    for doc in nlp.pipe(corpus_file):
        spans = list()
        for p in doc.noun_chunks:
            spans.append(p)
        i += 1
        if len(spans) > 0:
            span = spans.pop(0)
        else:
            span = None
        spanWritten = False
        for token in doc:
            if span is None:
                if len(token.text.strip()) > 0:
                    marked_corpus_file.write(token.text + ' ')
            else:
                if token.idx < span.start_char or token.idx >= span.end_char:  # outside a span
                    if len(token.text.strip()) > 0:
                        marked_corpus_file.write(token.text + ' ')
                else:
                    if not spanWritten:
                        # normalize NP
                        np = span.text
                        if np not in np2count:
                            np2count[np] = 1
                        else:
                            np2count[np] += 1
                        norm = spacy_normalizer(np, span.lemma_)
                        np2id[np] = norm
                        if norm not in id2rep:
                            id2rep[norm] = np
                        if norm in id2group:
                            if np not in id2group[norm]:
                                id2group[norm].append(np)
                            elif np2count[np] > np2count[id2rep[norm]]:
                                id2rep[norm] = np  # replace rep
                        else:
                            id2group[norm] = [np]
                            id2rep[norm] = np
                        # mark NP's
                        text = id2rep[norm].replace(' ', args.mark_char) + args.mark_char
                        marked_corpus_file.write(text + ' ')
                        spanWritten = True
                    if token.idx + len(token.text) == span.end_char:
                        if len(spans) > 0:
                            span = spans.pop(0)
                        else:
                            span = None
                        spanWritten = False
        marked_corpus_file.write('\n')
        if i % 500 == 0:
            logger.info('%i of %i lines', i, num_lines)

# write grouping data :

    corpus_name = os.path.basename(args.corpus)
    with open(
            'id2group_' + corpus_name + '_' + str(
                datetime.datetime.now().time()
            ), 'w', encoding='utf8') as id2group_file:
        id2group_file.write(json.dumps(id2group))

    with open(
            'id2rep_' + corpus_name + '_' + str(
                datetime.datetime.now().time()
            ), 'w', encoding='utf8') as id2rep_file:
        id2rep_file.write(json.dumps(id2rep))

    with open(
            'np2id_' + corpus_name + '_' + str(
                datetime.datetime.now().time()
            ), 'w', encoding='utf8') as np2id_file:
        np2id_file.write(json.dumps(np2id))

    corpus_file.close()
    marked_corpus_file.flush()
    marked_corpus_file.close()
