# Компоненты и смеси
**Назначение:** показать, как создавать компоненты из семейств `pysatl-core`, фиксировать параметры и собирать из компонент начальную смесь.

Примеры продолжают [быстрый старт](quickstart.md): используются его импорты, функция `show` и выборка `X`.

## Семейства и параметризации

Компонента смеси — распределение `ParametricFamilyDistribution` из `pysatl-core`. Своих распределений в `pysatl-mpest` нет, поэтому подходит любое семейство, зарегистрированное в `pysatl-core`. Сейчас там есть:

| `FamilyName` | Параметризации | Параметры |
|---|---|---|
| `NORMAL` | `meanStd` (основная), `meanVar`, `meanPrec`, `exponential` | `mu`, `sigma` в `meanStd`; `mu`, `var` в `meanVar` |
| `EXPONENTIAL` | `rate` (основная), `scale` | `lambda_` в `rate`; `beta` в `scale` |
| `CONTINUOUS_UNIFORM` | `standard` (основная), `meanWidth`, `minRange` | `lower_bound`, `upper_bound` в `standard` |

Без `parametrization_name` компонента создаётся в основной параметризации. Другая параметризация задаётся явно:

```python
exponential = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

a = normal(mu=0.0, sigma=1.0)                                  # meanStd
b = normal(mu=0.0, var=1.0, parametrization_name="meanVar")
c = exponential(lambda_=1.0)                                    # rate
d = exponential(beta=1.0, parametrization_name="scale")
```

Оценка работает в той параметризации, в которой создана компонента, и в ней же возвращает результат. От параметризации зависит скорость: для (`Normal`, `meanStd`) и (`Exponential`, `rate`) новые параметры на M-шаге вычисляются в явном виде, для остальных пар их ищет оптимизатор. Таблица всех случаев — в разделе [Стратегии M-шага](../math/strategies/index.md).

## Фиксированные параметры

Если часть параметров известна, их фиксируют через `family.view(...)`. Компонента из такого представления оценивает только свободные параметры:

```python
fixed_start = MixtureModel([normal.view(mu=0.0)(sigma=2.0), normal(mu=4.0, sigma=2.0)], weights=[0.5, 0.5])
show(em.fit(X, fixed_start))
```

```text
weight=0.406  {'sigma': 0.98}
weight=0.594  {'mu': 5.984, 'sigma': 1.489}
```

`component.parameters` перечисляет только свободные параметры, поэтому `mu` первой компоненты в выводе нет; он остаётся равным `0.0`. Фиксированный параметр нельзя назвать в `parameter_groups` у `ECM` и `ECME`: `fit` бросит `ValueError`.

## Смеси разных семейств

Компоненты одной смеси могут быть из разных семейств:

```python
mixed_truth = MixtureModel([exponential(lambda_=1.0), normal(mu=8.0, sigma=1.0)], weights=[0.3, 0.7])
Y = mixed_truth.generate(size=2000, random_state=1)

mixed_start = MixtureModel([exponential(lambda_=0.5), normal(mu=6.0, sigma=2.0)])
show(em.fit(Y, mixed_start))
```

```text
weight=0.295  {'lambda_': 1.042}
weight=0.705  {'mu': 7.988, 'sigma': 1.025}
```

Без `weights` компоненты получают равные веса.

## Работа со смесью

`MixtureModel` неизменяема. Методы `with_component`, `with_components`, `with_log_weights`, `with_added_component` и `without_component` возвращают новую смесь. Кроме того, у смеси есть:

- `weights`, `log_weights`, `components`, `n_components`, индексация `mixture[j]` и обход `for component in mixture`;
- `pdf`, `lpdf` и `loglikelihood` — плотность, её логарифм и логарифм правдоподобия выборки;
- `generate(size, random_state)` — выборка из смеси;
- `to_core_mixture()` — та же смесь как `FiniteMixtureDistribution` из `pysatl-core`, если дальше нужны его инструменты.

Две смеси равны (`==`), если у них одинаковые компоненты и веса, независимо от порядка компонент. Подробнее об устройстве `MixtureModel` — на странице модуля [`core`](../design/modules/core/core.md).
