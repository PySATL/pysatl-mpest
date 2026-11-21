# Быстрый старт
**Назначение:** на одном примере показать весь путь от выборки до оценённой смеси.

## Пример

Оценим смесь двух нормальных распределений по выборке из 2000 наблюдений:

```python
from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import FamilyName

from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import EM
from pysatl_mpest.estimators.iterative import LikelihoodBreakpointer, StepBreakpointer
from pysatl_mpest.optimizers import ScipyNelderMead

normal = ParametricFamilyRegister.get(FamilyName.NORMAL)


def show(mixture: MixtureModel) -> None:
    for weight, component in zip(mixture.weights, mixture):
        parameters = {name: round(value, 3) for name, value in component.parameters.items()}
        print(f"weight={weight:.3f}  {parameters}")


# Выборка из известной смеси.
truth = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=6.0, sigma=1.5)], weights=[0.4, 0.6])
X = truth.generate(size=2000, random_state=0)

# Начальное приближение: число компонент и их семейства известны, параметры угаданы грубо.
start = MixtureModel([normal(mu=1.0, sigma=2.0), normal(mu=4.0, sigma=2.0)], weights=[0.5, 0.5])

em = EM(
    breakpointers=[LikelihoodBreakpointer(1e-6), StepBreakpointer(200)],
    pruners=[],
    optimizer=ScipyNelderMead(),
)
result = em.fit(X, start)
show(result)
print(len(em.history))
```

Вывод:

```text
weight=0.405  {'mu': -0.052, 'sigma': 0.972}
weight=0.595  {'mu': 5.978, 'sigma': 1.495}
14
```

## Что произошло

1. `ParametricFamilyRegister.get(FamilyName.NORMAL)` вернул семейство нормальных распределений из `pysatl-core`. Вызов семейства с параметрами создаёт компоненту. Импорт `pysatl_mpest` регистрирует встроенные семейства `pysatl-core`, поэтому отдельно настраивать реестр не нужно.
2. `MixtureModel` объединил компоненты и веса. Смесь неизменяема: `fit` не меняет `start`, а возвращает новую смесь.
3. `EM` повторял итерации, пока не сработал один из критериев остановки: изменение логарифма правдоподобия меньше `1e-6` или 200 итераций. Здесь сработал первый критерий после 14 итераций.
4. Для нормального распределения в параметризации `meanStd` новые параметры вычисляются в явном виде, и `ScipyNelderMead` не понадобился. Оптимизатор обязателен всё равно: он нужен для семейств и параметризаций без решения в явном виде.

Оценки близки к истинным параметрам `truth`. Совпадать точно они не обязаны: оценка строится по конечной выборке.

## Что дальше

- [Компоненты и смеси](mixtures.md) — другие семейства и параметризации, фиксированные параметры, смеси разных семейств.
- [Выбор и настройка оценщика](estimators.md) — чем различаются `EM`, `ECM` и `ECME`, критерии остановки, удаление компонент и оптимизаторы.
- [Результат и история](results.md) — история итераций, ошибки и типичные проблемы оценки.
