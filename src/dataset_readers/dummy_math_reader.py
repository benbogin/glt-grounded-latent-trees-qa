import random
from typing import Tuple, List, Set

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, MetadataField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from overrides import overrides


@DatasetReader.register("dummy_math")
class DummyMathReader(DatasetReader):
    def __init__(self, epoch_size: int, validation_epoch_size: int,
                 train_max_num_numbers=8,
                 len_split_num_numbers=10,
                 max_answer=50,
                 random_seed=None,
                 randomly_pad=False,
                 randomly_pad_end=False,
                 split_by_operation_pos=False,
                 split_by_operation_order_num_keys=10,
                 math_operations=("+", "*", "-", "//")
                 ):
        super().__init__(lazy=True)
        self._epoch_size = epoch_size
        self._validation_epoch_size = validation_epoch_size
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._iid_validation = set()
        self._op_split_validation = set()
        self._length_split_validation = set()
        self._train_max_num_numbers = train_max_num_numbers
        self._len_split_num_numbers = len_split_num_numbers
        self._max_answer = max_answer
        self._math_operations = math_operations

        self.random = random.Random(random_seed)

        self.split_by_operation_pos = split_by_operation_pos
        self.randomly_pad = randomly_pad
        self.randomly_pad_end = randomly_pad_end

        self.operation_pos = None
        numbers = list(range(train_max_num_numbers))
        self.random.shuffle(numbers)
        operation_pos = list(zip(numbers,math_operations*(len(numbers)//len(math_operations))+math_operations[:len(numbers)%len(math_operations)]))
        self.operation_pos = operation_pos[:split_by_operation_order_num_keys]

        # operation split
        for i in range(validation_epoch_size):
            problem_string, label, keys = self.choose_numbers(max_answer=max_answer,
                                                              min_length=train_max_num_numbers,
                                                              max_length=train_max_num_numbers,
                                                              override_operation_pos=self.operation_pos)
            self._op_split_validation.add((problem_string, label))

        # length split
        for i in range(validation_epoch_size):
            problem_string, label, keys = self.choose_numbers(max_answer=max_answer,
                                                              min_length=len_split_num_numbers,
                                                              max_length=len_split_num_numbers,
                                                              avoid_operation_pos=self.operation_pos)
            self._length_split_validation.add((problem_string, label))

        # iid split
        for i in range(validation_epoch_size):
            problem_string, label, keys = self.choose_numbers(max_answer=max_answer,
                                                              min_length=train_max_num_numbers,
                                                              max_length=train_max_num_numbers,
                                                              avoid_operation_pos=self.operation_pos)
            self._iid_validation.add((problem_string, label))

    @overrides
    def _read(self, file_path: str):
        count = 0
        is_train = 'train' in file_path
        if not is_train:
            for problem_string, label in self._iid_validation:
                yield self.text_to_instance(question=problem_string, label=str(label), dataset_type=0)
            for problem_string, label in self._op_split_validation:
                yield self.text_to_instance(question=problem_string, label=str(label), dataset_type=1)
            for problem_string, label in self._length_split_validation:
                yield self.text_to_instance(question=problem_string, label=str(label), dataset_type=2)
        else:
            while count < self._epoch_size:
                problem_string, label, key = self.choose_numbers(self._train_max_num_numbers, self._max_answer,
                                                                 avoid_operation_pos=self.operation_pos,
                                                                 avoid_questions=self._iid_validation)
                yield self.text_to_instance(question=problem_string, label=str(label))
                count += 1

    def choose_numbers(self, max_length: int, max_answer: int,
                       min_length: int = 2,
                       override_operation_pos: List[Tuple] = None,
                       avoid_operation_pos: List[Tuple] = None,
                       avoid_questions: Set = None,
                       ):
        answer = -1
        while answer < 0 or answer > max_answer:
            try:
                num_numbers = self.random.randint(min_length, max_length)
                all_numbers = list(range(10))

                possible_actions_per_pos = [list(self._math_operations) for _ in range(num_numbers-1)]

                if override_operation_pos:
                    for pos, action in override_operation_pos:
                        if pos >= len(possible_actions_per_pos):
                            continue
                        possible_actions_per_pos[pos] = [action]

                if avoid_operation_pos:
                    for pos, action in avoid_operation_pos:
                        if pos >= len(possible_actions_per_pos):
                            continue
                        possible_actions_per_pos[pos] = list(set(self._math_operations) - set([action]))

                actions = [self.random.choice(psb_actions) for psb_actions in possible_actions_per_pos]

                chosen_numbers = self.random.choices(all_numbers, k=num_numbers)

                problem_items = [str(item) for pair in zip(chosen_numbers, actions) for item in pair] + [str(chosen_numbers[-1])]

                problem_string = ' '.join(problem_items)
                answer = eval(problem_string)

                if avoid_questions is not None and (problem_string, answer) in avoid_questions:
                    continue
            except ZeroDivisionError:
                continue

        if self.split_by_operation_pos:
            key = set([(i, o) for (i, o) in enumerate(actions)])
        else:
            key = tuple(actions)
        return problem_string, answer, key

    def text_to_instance(self,
                         question: str,
                         label: str = None,
                         dataset_type: int = 0
                         ) -> Instance:
        if self.randomly_pad:
            pad_prefix = 0
            numbers = len(question.split())
            max_length = self._val_num_numbers * 2 - 1
            if numbers < max_length:
                to_pad = max_length - numbers
                pad_prefix = self.random.randint(0, to_pad)
                pad_suffix = to_pad - pad_prefix
                question = ('P ' * pad_prefix) + question

                if self.randomly_pad_end:
                    question += (' P' * pad_suffix)

        fields = {
            "tokens": TextField(tokens=[Token(str(n)) for n in question.split()], token_indexers=self._token_indexers),
            "answer": LabelField(label),
            "dataset_type": LabelField(dataset_type, skip_indexing=True),
            "metadata": MetadataField({
                "question": question.split(),
                "answer": label,
            })
        }

        if self.randomly_pad:
            fields['pad_start'] = IndexField(pad_prefix, fields['tokens'])

        return Instance(fields)
