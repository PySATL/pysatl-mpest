"""Module from generating datasets with the given sample size, experimental counts and mixture"""

import argparse
from json import JSONEncoder, dump
from random import uniform

import numpy
from scipy.stats import dirichlet

from mpest import Distribution, MixtureDistribution
from mpest.models import AModel, ExponentialModel, GaussianModel, WeibullModelExp

DISTS = {
    ExponentialModel().name: ExponentialModel,
    GaussianModel().name: GaussianModel,
    WeibullModelExp().name: WeibullModelExp,
}


class NumpyEncoder(JSONEncoder):
    """
    An encoder that solves the problem with numpy arrays that are not dumped in the JSON
    """

    def default(self, o):
        if isinstance(o, numpy.integer):
            return int(o)
        if isinstance(o, numpy.floating):
            return float(o)
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        return super().default(o)


class DataSetGenerator:
    """
    Class that generates datasets from mixture.
    You can select the sample size and the number of experiments.
    """

    def create_mixture(self, models: list[type[AModel]]):
        """Function for generating random mixture"""
        dists = []
        priors = dirichlet.rvs([1 for _ in range(len(models))], 1)[0]
        for m in models:
            if m == ExponentialModel:
                params = [uniform(0.1, 5)]
            elif m == GaussianModel:
                params = [uniform(-5, 5), uniform(0.1, 5)]
            else:
                params = [uniform(0.1, 5), uniform(0.1, 5)]

            dists.append(Distribution.from_params(m, params))

        return MixtureDistribution.from_distributions(dists, priors)

    def generate_dataset(
        self, exp_count: int, sample_size: int, models: list[type[AModel]]
    ):
        """
        Function that generate dataset.
        Outputs the json file with structure:
        [
            [
                {distribution_name: [params, prior], ...}
                [samples]
            ]
        ]

        :param exp_count: Count of experiments (The length of the outer list in the file)
        :param sample_size: Sample size
        :param models: list of classes of models.
        """

        dataset = []
        for _ in range(exp_count):
            mixture = self.create_mixture(models)
            distribution_dict = {}
            sample = mixture.generate(sample_size)
            for d in mixture:
                if d.model.name in distribution_dict:
                    distribution_dict[d.model.name] += [d.params] + [
                        d.prior_probability
                    ]
                else:
                    distribution_dict[d.model.name] = [d.params] + [d.prior_probability]

            dataset.append([distribution_dict, numpy.sort(sample)])

        module_names = "".join(d.model.name[0] for d in mixture)
        output_fname = f"ds_{module_names}_{sample_size}.json"
        with open(output_fname, "w+", encoding="UTF-8") as f:
            dump(dataset, f, cls=NumpyEncoder)

    def generate(self, exp_count, sample_size, model_names):
        """
        :param exp_count: Count of experiments (The length of the outer list in the file).
        :param sample_size: Sample size.
        :param models: list of model names.
        """
        models = [DISTS[name] for name in model_names]
        self.generate_dataset(exp_count, sample_size, models)


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser("Generate datasets")
    parser.add_argument("exp_count", type=int, help="Count of experiments")
    parser.add_argument("sample_size", type=int, help="Size of generated sample")
    parser.add_argument(
        "--m",
        nargs="+",
        choices=DISTS.keys(),
        help="Models which will be used in mixture",
    )

    items = parser.parse_args()
    exp_count, sample_size, model_names = items.exp_count, items.sample_size, items.m
    ds_generator = DataSetGenerator()
    ds_generator.generate(exp_count, sample_size, model_names)


if __name__ == "__main__":
    main()
