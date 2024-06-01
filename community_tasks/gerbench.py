# MIT License

# Copyright (c) 2024 The HuggingFace Team
# Copyright (c) 2024 Philip May, Deutsche Telekom AG

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
This module implements the 4 tasks of deutsche-telekom/Ger-RAG-eval.
See: https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Task 1: Sentence Errors
# Detenct Sentence in German
# The task is to decide which one is the correct sentence in german
task1 = LightevalTaskConfig(
    name="gerbench:sentence_errors",
    prompt_function="prompt_fn_sentence_errors",
    suite=["community"],
    hf_repo="ellamind/gerbench_sentence_errors",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=["loglikelihood_acc"],
    version=0,
)

# Task 2: Choose context by question.
# Given is a question and 4 contexts.
# The task is to decide which context can answer the question.
task2 = LightevalTaskConfig(
    name="gerbench:next_word",
    prompt_function="prompt_fn_next_word",
    suite=["community"],
    hf_repo="ellamind/gerbench_next_word",
    hf_subset=None,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=["loglikelihood_acc"],
    version=1,
)


def prompt_fn_sentence_errors(line, task_name: str = None):
    instruction = """\
Es sind vier deutschsprachige Sätze unter A, B, C und D gegeben. Drei enthalten einen kleinen Fehler und einer ist der Orirignalsatz. Bitte antworte mit dem Buchstaben (A, B, C oder D) des Satzes, der KEINEN Fehler enthält!"""

    query_template = """
A: {choice_a}
B: {choice_b}
C: {choice_c}
D: {choice_d}

Antwort:"""
    query = instruction + query_template.format(
        choice_a=line["answer1"],
        choice_b=line["answer2"],
        choice_c=line["answer3"],
        choice_d=line["answer4"],
    )
    choices = ["A", "B", "C", "D"]
    answer_mapping = {"answer1": "A", "answer2": "B", "answer3": "C", "answer4": "D"}
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(answer_mapping[line["golden"]]),
    )


def prompt_fn_next_word(line, task_name: str = None):
    instruction = """\
Bitte setze den folgenden deutschsprachiger Satz korrekt fort!"""

    query_template = """
{satz_bis_zum_letzten_wort} """
    query = instruction + query_template.format(
        satz_bis_zum_letzten_wort=line["satz_bis_zum_letzten_wort"],
    )
    choices = [line["answer1"], line["answer2"], line["answer3"], line["answer4"]]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(line[line["golden"]]),
    )


# STORE YOUR EVALS
_TASKS = [task1, task2]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
