"""A module containing classes for storing information about the results of estimating at the second stage"""

from collections.abc import Iterable, Iterator

import numpy as np

from experimental_env.preparation.dataset_description import DatasetDescrciption
from mpest import MixtureDistribution
from mpest.utils import ResultWithLog


class StepDescription:
    """
    A class containing information about each step of the algorithm
    """

    def __init__(self, mixture, time):
        self._result_mixture = mixture
        self._time = time

    @property
    def result_mixture(self) -> MixtureDistribution:
        """
        The property containing the mixture from this step.
        """
        return self._result_mixture

    @property
    def time(self) -> float:
        """
        A property containing the time that was spent to complete this step of the algorithm
        """
        return self._time

    def to_yaml_format(self) -> dict:
        """
        A function for converting information to yaml format
        """
        output = {}

        # Get params of distributions
        dists = []
        for d in self._result_mixture:
            prior = float(d.prior_probability) if d.prior_probability else np.nan
            dists.append(
                {
                    "type": d.model.name,
                    "params": d.params.tolist(),
                    "prior": prior,
                }
            )
        output["step_distributions"] = dists
        output["time"] = self._time

        return output


class ExperimentDescription(Iterable):
    """
    A class containing information about all the steps of the algorithm.
    """

    def __init__(self):
        self._init_mixture = None
        self._steps = []
        self._ds_descr = None
        self._error = False

        self._pointer = -1

    @classmethod
    def from_result(
        cls,
        init_mixture: MixtureDistribution,
        result: ResultWithLog,
        ds_descr: DatasetDescrciption,
    ):
        """
        A class method for creating an object from a result with type ResultWithLog
        """
        self = cls()
        self._init_mixture = init_mixture
        self._steps = [StepDescription(step.result.content, step.time) for step in result.log.log]
        self._ds_descr = ds_descr
        self._error = bool(result.log.log[-1].result.error)
        return self

    @classmethod
    def from_steps(
        cls,
        init_mixture: MixtureDistribution,
        steps: list[StepDescription],
        ds_descr: DatasetDescrciption,
        error: bool,
    ):
        """
        A class method for creating an object from a steps with type StepDescription
        """
        self = cls()
        self._init_mixture = init_mixture
        self._steps = steps
        self._ds_descr = ds_descr
        self._error = error
        return self

    @property
    def base_mixture(self):
        """
        The property for obtaining a base mixture
        """
        return self._ds_descr.base_mixture

    @property
    def init_mixture(self):
        """
        The property for obtaining an initial mixture
        """
        return self._init_mixture

    @property
    def samples(self):
        """
        The property for obtaining a samples.
        """
        return self._ds_descr.samples

    @property
    def samples_size(self):
        """
        The property for obtaining a sample size.
        """
        return self._ds_descr.samples_size

    @property
    def exp_num(self):
        """
        The property for obtaining an experiment number.
        """
        return self._ds_descr.exp_num

    @property
    def steps(self) -> list[StepDescription]:
        """
        A property that contains all the steps of the EM algorithm in this experiment
        """
        return self._steps

    @property
    def error(self):
        """
        A property to get information about whether there was an error during the execution of the algorithm or not.
        """
        return self._error

    def __next__(self):
        """
        Get next element.
        """
        return self._steps[0]

    def __iter__(self) -> Iterator:
        """
        Iterating over steps
        """
        return iter(self._steps)

    def to_yaml_format(self):
        """
        A function to convert the descriptor to yaml format for saving.
        """
        output = self._ds_descr.to_yaml_format()

        dists = []
        for d in self._init_mixture:
            dists.append(
                {
                    "type": d.model.name,
                    "params": d.params.tolist(),
                    "prior": float(d.prior_probability),
                }
            )
        output["init_distributions"] = dists

        output["error"] = self._error

        return output
