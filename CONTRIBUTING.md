# Pysatl-mpest project contributing guide

Thank you very much if you have decided to contribute to our project.

## How to Contribute to pysatl-mpest

Follow these steps to get started. We use the Forking Workflow model.

### 1. Fork the Repository

First, you need to fork (create a copy of) the main repository.

1. Navigate to the repository page: https://github.com/PySATL/pysatl-mpest
2. Click the "Fork" button in the top-right corner of the page. This will create a copy of the repository in your GitHub account.



### 2. Clone the Fork and Create a Branch

Now that you have a fork, clone it to your local machine:

```bash
git clone https://github.com/YOUR-USERNAME/pysatl-mpest.git
cd pysatl-mpest
```

Before you start making changes, create a new branch in following format `<your-username>/<what-this-branch-solves>` if this branch on the original repository. If you working in the fork, name your branches at your own discretion.

For example: `iraedeus/add-new-solver` or `vityana/memory-leak-issue`.

```bash
git checkout main
git checkout -b <your-username>/<what-this-branch-solves>
```



### 3. Set Up Your Development Environment

To start working on the code, you need to install the project dependencies and set up the pre-commit hooks. These hooks automatically check your code for quality and style before each commit.

1. **Install dependencies:**

   Follow steps in [README.md](README.md)

2. **Install pre-commit hooks (Mandatory):**

   ```bash
   pre-commit install
   ```

   This command sets up the hooks in your local Git repository. **This is a required step for all contributors.**

Now your environment is ready. The hooks will run automatically every time you run `git commit`.



### 4. Make and Commit Changes

Work on the code in your new branch. When you're ready to save your changes, create a commit. We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to standardize commit messages.

Use the following types:

- `feat`: to add new functionality.
- `fix`: to fix a bug in the project.
- `refactor`: for code refactoring, such as renaming a variable or improving readability.
- `test`: to add or refactor tests.
- `struct`: for changes related to the project structure (NOT CODE), for example, changing folder locations.
- `ci`: for various CI/CD tasks.
- `docs`: for changes in documentation.
- `chore`: for other changes that do not affect the code or tests (e.g., updating `.gitignore`, `README.md`).



```bash
# Add the files you have changed
git add .

# Create a commit with a proper message
git commit -m "feat(auth): implement user authentication"
```

Try to divide commits into atomic and independent parts. That is, do not add unrelated changes to the code in one commit.



### 5. Push Changes to Your Fork

After you have made one or more commits, push your branch to your remote fork on GitHub:

```bash
git push
```



### 6. Documentation requirements

High-quality documentation is crucial for the project's maintainability and usability. All contributions that add or modify code **must** include corresponding documentation updates.

#### File Header and Module Docstring

Every Python source file (`.py`) must begin with a standard header containing metadata and a module-level docstring. This ensures consistency and proper attribution.

**Template:**

```python
"""Module docstring"""

__author__ = "Your Name"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

# ... rest of the module code
```

#### Docstring Style and Content

We use the **NumPy docstring standard** as our base format. All public modules, classes, methods, and functions must have comprehensive docstrings that are compatible with **Sphinx** for automatic documentation generation.

**Key Requirements:**

1. **Use Sphinx Roles for Cross-Referencing:**
   To ensure that our documentation is fully navigable, all references to other code elements (classes, methods, attributes, parameters) within docstrings **must** use Sphinx roles. This turns them into clickable hyperlinks in the generated documentation.
   - Use ``:class:`ClassName` `` to reference a class.
   - Use ``:meth:`method_name` `` to reference a method.
   - Use ``:attr:`attribute_name` `` to reference an attribute or property.
   - Use ``:func:`function_name` `` to reference a function.
   - To shorten a long path, use a tilde (~). For example, ``:class:~pysatl_mpest.core.mixture.MixtureModel` `` will be rendered simply as `MixtureModel`.
2. **Include autosummary for Public APIs:**
   For classes, the docstring should include an `.. autosummary::` block. This directive instructs Sphinx to automatically generate a summary table of the class's public methods and attributes.
   - List all public methods that should be included in the documentation.
   - Use the `:toctree: generated/` option.
3. **Structure and Detail:**
   - **Classes:** The docstring should clearly describe the class's purpose, followed by Parameters (for the constructor), Attributes, Methods (via autosummary), and Notes or Examples where applicable.
   - **Methods/Functions:** Must include a description, Parameters with types, Returns with type, and Raises for any exceptions thrown.

**Comprehensive Example of a Class Docstring:**

```python
class Pipeline(BaseEstimator):
    """An estimator that fits a mixture model via a configurable iterative process.

    The pipeline executes a sequence of defined steps in a loop. After each full
    sequence of steps, pruning strategies are applied, and stopping conditions
    are checked. The process is orchestrated by :class:`.PipelineStep`,
    :class:`.Breakpointer`, and :class:`.Pruner` components.

    Parameters
    ----------
    steps : Sequence[PipelineStep]
        An ordered sequence of steps to be executed in each iteration.
    breakpointers : Sequence[Breakpointer]
        A sequence of strategies that define the stopping conditions.

    Attributes
    ----------
    steps : list[PipelineStep]
        The ordered list of operations to be performed in each iteration.
    logger : PipelineLogger
        An object that collects information about each iteration.

    Raises
    ------
    ValueError
        If the sequence of :attr:`steps` or :attr:`breakpointers` is empty.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    # ... class implementation
```



#### Building and Previewing Documentation Locally

Our documentation is built using **Sphinx**. Before submitting your pull request, you **must** generate the documentation locally to ensure there are no formatting errors, warnings, or broken links.

1. Navigate to the `docs` directory:

   ```bash
   cd docs
   ```

2. Build the documentation:

   ```bash
   make clean html
   ```

3. Preview the result:

   The output will be generated in the `docs/build/html` directory. Open `docs/build/html/index.html` in your web browser to review the changes and verify that everything renders correctly.

#### Updating Architectural Documentation

If your changes introduce significant architectural modifications (e.g., adding a new major component, changing the core data flow, or altering the relationship between modules), you must also update the design documentation. This includes updating both the descriptive text and any relevant diagrams (e.g., UML, flowcharts).



### 7. Testing Requirements

To maintain code quality and prevent regressions, all contributions must be accompanied by comprehensive tests. We use `pytest` for running tests, `pytest-cov` to enforce a minimum code coverage standard, and `mutmut` to check that the tests notice wrong results. The tests are the primary specification of the behaviour of the code: by the layout of the files and the names of the tests you should be able to tell where a behaviour is checked and where a new check belongs.

- All new functionality (`feat`) must be accompanied by tests that cover its behavior.
- A bug fix must be accompanied by a test that fails without the fix.

#### Unit and Integration Tests

The level of a test is defined by its dependencies, not by the size of the tested component.

- **Unit tests** (`tests/unit/`) check one class or function. Its behavioural dependencies (pipeline steps, optimizers, strategies, pruners, breakpointers) are replaced with test doubles. Real `MixtureModel` objects, `pysatl-core` distributions and `PipelineState` are allowed as input data, as long as the test does not execute their behaviour as part of what it checks. A real mixture may also be written to by the class under test; then assert what the class under test did and do not re-check behaviour of the mixture itself (for example, the normalization of weights), which is specified in the tests of `MixtureModel`.
- **Integration tests** (`tests/integration/`) check that real components of `pysatl_mpest` and `pysatl-core` work together. They contain **no mocks, spies or scripted replacements** of components.
- Replacing a numeric constant with `monkeypatch` (for example, a threshold set to `0.25` to test a boundary exactly) is a test parameter, not a mock, and does not change the level of the test. Pass such a value explicitly, through a parameter or an explicitly requested fixture, and never through a hidden `autouse` fixture.
- A value that a strategy hands to the optimizer, for example the objective function of the Q-function or the likelihood strategy, can be observed only through an optimizer double, so such a test is a unit test. Integration tests of strategies check the final result with a real optimizer.

#### Where Tests Live

The layout of `tests/unit/` and `tests/integration/` mirrors the layout of `pysatl_mpest/`.

- **One file per module that has its own behaviour**, named `test_<module>.py`; the underscore of a private module is kept (`core/_sampling.py` is tested in `core/test__sampling.py`). Abstract base classes, re-exports in `__init__.py`, `typings.py` and empty or reserved packages have no test file: their contract is checked through the concrete implementations. The root `pysatl_mpest/__init__.py` is an exception: it registers the built-in families, so it is tested in `tests/integration/test___init__.py`.
- **Every behaviour has one owner**, the module that implements it, and is specified in the tests of that module. Other tests only check how the owner is used. For example, shifting the indices of the optimization blocks after pruning is specified in the tests of `MaximizationStep`; the tests of `Pipeline` check only that every step is notified about the removed components. Likewise, the equality of mixtures is specified in the tests of `MixtureModel`. The unit tests of the owner specify the whole contract; an integration test may confirm it on real components (for example, that a mixture of plain components equals a mixture of the same components with fixed parameters), but does not repeat the full table of cases.
- **An integration test is placed by the entry point of the chain** it exercises (for example, a complete fit with `ECM` belongs to the tests of `ECM`). If a scenario has several entry points, choose the main one and mention the others in a comment.
- **`PipelineState` has no test file of its own.** It is checked through the steps and the `Pipeline` that use it. Tests of a repeated `fit` of the same object belong to the tests of `Pipeline` (its state), with at most one end-to-end scenario in the tests of `ECM` and of `ECME`; what such a `fit` must reset is not settled yet (see Contested Contracts).

To find an existing check, find the module that owns the behaviour, open its test file at the same relative path under `tests/unit/` or `tests/integration/`, and look for a test whose name states that behaviour.

| Behaviour | Owner | Level |
|---|---|---|
| Shifting block indices after pruning | `MaximizationStep` | unit |
| Notifying the steps about pruned components | `Pipeline` | unit |
| Normalization of the mixture weights and equality of mixtures | `MixtureModel` | unit |
| Closed-form Q-function of the normal distribution | `q_function` strategy | integration |
| Complete fit of a two-component mixture | `ECM` | integration |

#### Parametrization

Contracts that depend on the support or the tails of a distribution are parametrized over these properties, not over family names:

| | Light tails | Heavy tails |
|---|---|---|
| Infinite support | `normal` | no representative |
| Semi-infinite support | `exponential` | no representative |
| Finite support | `uniform` | not applicable |

- Parametrize over the cells that have a representative. Do not create fake distributions for the empty cells; when `pysatl-core` gets a suitable family, add it to its cell.
- Contracts of this kind are, for example, points outside the support in the E-step, the penalty in the observed-data likelihood, the numerical Q-function, and `lpdf` and `generate` at the boundary of the support. Use a cell only where the contract makes sense for it: a point outside the support exists for `exponential` and `uniform` but not for `normal`, where a point far in the tail takes its place.
- A closed-form formula (`normal` with `meanStd`, `exponential` with `rate`) is tested by its own tests for the pair of the family and the parametrization it is registered for.
- A contract common to the strategies of one dispatcher is parametrized over all pairs of a family and a parametrization registered in it. For example, the Q-function and moments strategies raise `ValueError` when `H` is `None`, while the observed-data likelihood does not use `H`.

#### Fixtures, Helpers and Test Doubles

- **Share doubles and assertion helpers only when two test files use them.** A test double goes to `tests/mocks/`, and an assertion helper to `tests/helpers/`, only if at least two test files use it. A double or helper with one user stays in that file. Remove shared code that nobody uses.
- **Factories and fixtures follow the same idea.** One that a single file needs stays in that file; one that several files need goes to `tests/helpers/` (a factory function) or to the nearest `conftest.py` (a fixture), for example the sample and the starting mixture shared by the integration tests of `ECM` and `ECME` in `tests/integration/estimators/conftest.py`.
- **Name local doubles by what they do** (`RemovingFirstComponentPruner`, `StopOnCheck`), and never reuse the name of a shared double.
- **Keep doubles close to the real API.** A double implements the part of the interface the test needs, with the same signatures and return types as the real class. A double may subclass the real class, be created with `create_autospec`, or implement the used part of the interface by hand.
- **Fixtures** have function scope and are requested explicitly by the test. `autouse` is for the settings of the whole environment that do not change a mathematical result (the global seed in `tests/conftest.py`); it must not hide a substitution of a constant or an object under test.
- **Randomness.** The global fixture fixes the seed to 42, and Hypothesis runs with `derandomize=True`. Create random data in a test with an explicit generator, for example `rng = np.random.default_rng(42)`.
- **Helpers create valid objects.** A helper must return an object that satisfies the production type annotations; do not pass `None` with `# type: ignore` where a mixture is required, and do not pass `optimizer=None` to a strategy. Pass a double in a unit test and a real optimizer in an integration test.

#### Naming

- A test name states a behaviour: `test_rejects_removal_of_last_component`, not `test_step` or `test_case_1`.
- Give parametrized cases readable `ids` (`normal`, `uniform`).
- Keep tests as module-level functions. If a file covers three or more separate topics (a topic is a group of related contracts, for example construction, `fit`, pruning), you may group the tests of each topic in a `Test<Topic>` class named after the topic. Do not use classes for one or two topics.
- Put every test in the class of its own topic, and do not repeat the topic in the test name: `TestClearAfterPrune::test_shifts_blocks_after_removed_component`, not `TestClearAfterPrune::test_clear_after_prune_shifts_blocks`.
- Keep Hypothesis tests (`@given`) as module-level functions, also in a file with classes; place such a test right after the class of its topic. Hypothesis fails a `@given` method that is called from different instances of its class (`HealthCheck.differing_executors`), and this happens when `mutmut` runs the suite several times in one process.

#### Adding or Changing a Test

1. Formulate the contract: for which input what result, state change or exception is expected.
2. Find the owner of the behaviour and the tests that already check it or its neighbours.
3. Choose one of three actions: **no new test** if the contract is already checked; **extend** an existing test or parametrization if only the input differs (a new case, a new cell of the table above); **add a separate test** if it is a different behaviour.
4. For a bug fix, write the test first and make sure it fails without the fix.
5. Name the test by the behaviour and put it in the file of the owner.
6. Check whether the new test makes an older, weaker one redundant, and remove the redundant one.
7. Do not mix unrelated changes of neighbouring tests into the same commit.
8. Run the affected tests first, then the entire suite.

#### Contested Contracts

If the behaviour of the code looks doubtful or is not documented, do not fix your assumption in a test. Describe it in an issue or in the description of your pull request and agree on it with the maintainers first. When a test checks an error, assert the exception class and a substring of the message (`pytest.raises(ValueError, match="...")`); assert the exact text only if the docstring declares it. For example, what a repeated `fit` of the same estimator must reset (the history, the optimization blocks, the state of the breakpointers) is not settled; check only the parts that are documented.

#### Running Tests and Checking Coverage

Before creating a pull request, you **must** run the entire test suite locally. Our project is configured to automatically check test coverage on every run.

**Run all tests and generate a coverage report with this single command:**

```bash
poetry run pytest
```

This command will:

1. Execute all tests within the project.
2. Display a code coverage summary in the terminal, highlighting any files or lines that are not covered by tests.
3. Generate a detailed, interactive HTML report in a new `htmlcov/` directory.

To visually analyze which lines of your code are not covered, open the `htmlcov/index.html` file in your browser. This is an excellent way to find gaps in your tests.

**Important:** Our CI pipeline will fail if your contribution causes the total code coverage to drop below the required threshold (80%). Ensure that all new code is adequately tested.

During development run only the tests that your change can affect:

```bash
poetry run pytest -o addopts= tests/unit/estimators/test_ecm.py -k "fit"
```

**Do not run `pytest` with other options from the repository root without `-o addopts=`.** The `addopts` in `pyproject.toml` turn on `pytest-cov`, and any such run, including `pytest --help`, overwrites the `.coverage` file of your last full run. The override `-o addopts=` switches the coverage options off.

#### Mutation Testing

Code coverage only shows which lines the tests execute, not whether they would notice a wrong result. Mutation testing checks that: [`mutmut`](https://github.com/boxed/mutmut) makes a small change to the code (flips a comparison, removes an argument, alters a constant), runs the tests, and reports the change as *killed* if a test fails or as *survived* if every test still passes. A survivor is either a missing test or a change that does not alter the behaviour (an *equivalent* mutant).

It is run by hand and is not part of CI. The modules that are mutated are listed in the `[tool.mutmut]` section of `pyproject.toml`: the mixture model, the component helpers, the strategies of the maximization step, both steps of the iterative pipeline, the optimization block and the pipeline itself. A run takes about two minutes.

```bash
poetry run mutmut run                       # generate and test the mutants
poetry run mutmut results                   # list the survivors
poetry run mutmut show <mutant name>        # show what a survivor changes
```

The results are stored in the `mutants/` directory, which is ignored by git. If you change which functions are decorated, or add modules, delete it (`rm -rf mutants`) before the next run: otherwise mutants of the new code can be reported as having no tests, because `mutmut` selects the tests for a function from statistics gathered earlier.

For every survivor either add a test that kills it, or check that it is equivalent, that is, that the change cannot alter any behaviour (for example, it removes an argument that is equal to its default), and leave it. When you change a mutated module, run `mutmut` and make sure that you do not add survivors that are not equivalent.

**Limitations of the tool that shape the code**

- `mutmut` does not mutate functions that have a decorator. That is why the implementations of the strategies in `q_function.py` and `moments.py` are plain functions registered by a call (`q_function_strategy.register(...)(function)`) and not by `@...register(...)`. Keep it that way, otherwise the most important numerical code silently drops out of the check. Properties are not mutated either.
- The thresholds of the analytical strategies (`NUMERICAL_TOLERANCE`) are tested exactly at the boundary by replacing the constant with `0.25`, which is exact in binary floating point (in the integration tests of the strategies, with the constant replaced explicitly).

### 8. Create a Pull Request (PR)

Once your branch with the changes is in your fork, you can create a Pull Request to propose your changes to the main repository.

1. Go to your fork's page on GitHub (https://github.com/YOUR-USERNAME/pysatl-mpest).
2. You will see a notification prompting you to create a Pull Request for your recently pushed branch. Click the **"Compare & pull request"** button.
3. Ensure the base repository is `PySATL/pysatl-mpest` with the main branch, and the head repository is your fork and your working branch.
4. Give your Pull Request a meaningful title (following the [Conventional Commits](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.conventionalcommits.org%2F) standard) and add a detailed description of the changes you've made.

#### Rules for Your PR to be Merged

For your Pull Request to be merged, it must meet the following requirements configured in the repository:

- **CI/CD Checks Must Pass:** All automated status checks must complete successfully. This includes linters, the entire test suite, and a **minimum code coverage check**. If any check fails, you will need to fix the issue and push new commits to your branch.
- **Mandatory Code Review:**
  - Your PR must receive at least **one approval** from another project member.
  - The PR author **cannot** approve their own work. The most recent set of changes must be approved by someone other than the person who pushed them.
  - If you push new commits to your branch after receiving an approval, that approval will be automatically dismissed. The changes will need to be reviewed again. This ensures that the latest version of the code is always reviewed.
- **All Conversations Must Be Resolved:** All comments and conversations on the code left during the review must be marked as resolved before the PR can be merged.
- **No Force Pushing:** The main branch is protected from force pushes to ensure a stable and clean commit history.

Each pull request must be reviewed by one of the maintainers:

* Danil Totmyanin ([iraedeus](https://github.com/iraedeus))
