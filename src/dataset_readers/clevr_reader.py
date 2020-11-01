import json
import logging
import os
from typing import List

import h5py
import numpy as np
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, MetadataField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__)


@DatasetReader.register("clevr")
class ClevrReader(DatasetReader):
    def __init__(self,
                 data_dir: str,
                 lazy: bool = False,
                 limit: int = None,
                 remove_stop_words: bool = False,
                 split_coreferences: bool = False,
                 load_closure_set: str = None,
                 fine_tune_closure_set: str = None
                 ):
        super().__init__(lazy=lazy)
        self._limit = limit
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._data_dir = data_dir
        self._remove_stop_words = remove_stop_words
        self._split_coreferences = split_coreferences
        self._fine_tune_closure_set = fine_tune_closure_set
        self._load_closure_set = load_closure_set

        self._cache_img_features = {}

    @overrides
    def _read(self, file_path: str):
        # get name of split (train/val/test) for features file (we make an assumption about the input file name...)
        if 'train' in file_path:
            split = 'train'
        elif 'test' in file_path:
            split = 'test'
        else:
            split = 'val'

        h5_file = f'bottomup_{split}_features.h5'
        f_feats = h5py.File(os.path.join(self._data_dir, h5_file), 'r')

        n_examples = 0

        file_inputs = [file_path]
        if self._load_closure_set:
            if self._load_closure_set and split == 'val':
                file_inputs.append(file_path.replace('val', f'closure_{self._load_closure_set}'))

        for file_path in file_inputs:
            is_ood_example = 'closure' in file_path
            with open(file_path, 'r') as f:
                lines = f
                limit = self._limit

                for i, line in enumerate(lines):
                    ex = json.loads(line)
                    n_examples += 1

                    tokens = [t.lower() for t in ex['question_tokens']]

                    if self._remove_stop_words:
                        stop_words = {'?', '!', 'the', 'there', 'is', 'a', 'as', 'it', 'of', 'are', 'its', 'other',
                                      'on', 'that'}
                        if self._split_coreferences:
                            stop_words.remove('it')
                            stop_words.remove('its')
                        tokens = [t for t in tokens if t not in stop_words]

                    # if configured, split sentence into two parts when ';' is found
                    tokens_sent_part2 = None
                    if self._split_coreferences:
                        if ';' in tokens:
                            delimiter_loc = tokens.index(';')
                            tokens, tokens_sent_part2 = tokens[:delimiter_loc], tokens[delimiter_loc + 1:]

                    yield self.text_to_instance(
                        i,
                        f_feats['features'][ex['image_index']],
                        f_feats['boxes'][ex['image_index']],
                        ex['question'],
                        tokens,
                        tokens_sent_part2,
                        ex['answer'] if 'answer' in ex else None,
                        ex['image_filename'],
                        is_ood_example,
                        ex['question_family_index'] if 'question_family_index' in ex else None,
                        ex['closure_group'] if 'closure_group' in ex else None,
                        ex['alternative_answer'] if 'alternative_answer' in ex else None,
                    )
                    if 0 < limit < i:
                        break

            logger.info(f"Total examples in {file_path}: {n_examples}")

    def text_to_instance(self,
                         question_index,
                         img_features,
                         img_boxes,
                         question: str,
                         question_tokens: List[str],
                         question_tokens_sent2: List[str] = None,
                         answer: str = None,
                         image_filename: str = None,
                         is_ood_example: bool = False,
                         question_family_index: int = False,
                         closure_group: str = None,
                         alternative_answer: str = None
                         ) -> Instance:

        if image_filename not in self._cache_img_features:
            n_boxes = int(sum([sum(box) > 0 for box in img_boxes]))
            img_features = np.array(img_features[:n_boxes], dtype=np.float32)
            img_boxes = np.array(img_boxes[:n_boxes], dtype=np.float32)
            self._cache_img_features[image_filename] = img_boxes, img_features, n_boxes

        boxes, features, n_boxes = self._cache_img_features[image_filename]

        tokens = [Token(t) for t in question_tokens]

        fields = {
            "tokens": TextField(tokens=tokens, token_indexers=self._token_indexers),
            "img_features": ArrayField(features),
            "patches_pos": ArrayField(np.array(boxes)),
            "n_img_features": LabelField(n_boxes, skip_indexing=True),
            "metadata": MetadataField({
                "question": question,
                "question_index": question_index,
                "question_tokens": question_tokens,
                "question_tokens_sent2": question_tokens_sent2,
                "answer": answer,
                "image_filename": image_filename,
                "pos_info": boxes,
                "question_family_index": question_family_index,
                "closure_group": closure_group,
                "alternative_answer": alternative_answer,
            }),
            "is_ood_example": LabelField(int(is_ood_example), skip_indexing=True),
        }

        if answer is not None:
            fields["answer"] = LabelField(answer) if answer is not None else None
            fields["alternative_answer"] = LabelField(alternative_answer) if alternative_answer else LabelField(answer)

        if self._split_coreferences:
            if not question_tokens_sent2:
                question_tokens_sent2 = []
            question_tokens_sent2 = [Token(t) for t in question_tokens_sent2]
            fields['tokens_sent2'] = TextField(tokens=question_tokens_sent2,
                                               token_indexers=self._token_indexers)

        return Instance(fields)
