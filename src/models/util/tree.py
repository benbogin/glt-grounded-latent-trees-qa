from typing import List

import torch


def sentence_to_tree(question: List[str], attention_probs: List[torch.Tensor], batch_index: int):
    def sentence_to_tree_rec(i: int, j: int):
        if i == 0:
            return question[j]
        if i == 1:
            return '(' + question[j] + ' ' + question[j + 1] + ')'

        probs = attention_probs[i][batch_index, j]
        best_split_num = probs.argmax()
        return '(' + sentence_to_tree_rec(best_split_num, j) + ' ' + sentence_to_tree_rec(i - 1 - best_split_num,
                                                                                          j + 1 + best_split_num) + ')'

    n = len(question)
    return sentence_to_tree_rec(n - 1, 0)


def sentence_to_tree_svgling(question: List[str], attention_probs, denotations, control_gates):
    control_name = ["-", "⋂", "⋃", "👁"]

    def sentence_to_tree_rec(i: int, j: int):
        phrase_denotations = [obj_id + 1 for (obj_id, d) in enumerate(denotations[i][j]) if d > 0.6]
        if not phrase_denotations:
            phrase_denotations = "𝜙"
        if i == 0:
            return (str(phrase_denotations) + "\n" + question[j],)

        probs = attention_probs[i][j]
        control = control_gates[i][j]
        best_split_num = probs.argmax()
        best_control = control_name[control.argmax()]
        phrase = (str(phrase_denotations) + ' ' + best_control, sentence_to_tree_rec(best_split_num, j),
                  sentence_to_tree_rec(i - 1 - best_split_num,
                                       j + 1 + best_split_num))
        return phrase

    n = len(question)
    return sentence_to_tree_rec(n - 1, 0)
