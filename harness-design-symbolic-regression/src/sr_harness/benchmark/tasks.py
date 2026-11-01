"""Two views of a task.

SymbolicRegressionTask  full task, including the hidden held-out data:
                        ID (new points inside the training range) and
                        OOD (out of distribution: points beyond the training range).
                        Lives only in the benchmark and evaluation layers.
TaskView                what a harness receives: description, variables and
                        training data. It has no held-out arrays at all, so a
                        harness cannot read them even by accident.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskView:
    task_id: str
    description: str
    target_name: str
    target_description: str
    variable_names: tuple
    variable_descriptions: tuple
    x_train: np.ndarray          # (n, d)
    y_train: np.ndarray          # (n,)


@dataclass
class SymbolicRegressionTask:
    task_id: str
    domain: str
    description: str
    target_name: str
    target_description: str
    variable_names: list
    variable_descriptions: list
    x_train: np.ndarray          # what the agent searches on (train R^2)
    y_train: np.ndarray
    x_train_dist: np.ndarray     # hidden: new points from the training range (ID R^2)
    y_train_dist: np.ndarray
    x_ood: np.ndarray | None     # hidden: points beyond the training range (OOD R^2)
    y_ood: np.ndarray | None
    ground_truth: str | None = None   # hidden: for analysis only

    def view(self) -> TaskView:
        return TaskView(self.task_id, self.description, self.target_name, self.target_description,
                        tuple(self.variable_names), tuple(self.variable_descriptions),
                        self.x_train.copy(), self.y_train.copy())
