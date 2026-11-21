# Design
**Назначение:** дать разработчику общее представление об устройстве библиотеки, используемом стеке, статусе возможностей и навигации по архитектурной документации.

```{toctree}
:hidden:
:maxdepth: 1

domain
architecture
behavior
modules/index
```

## Что делает библиотека

`pysatl-mpest` оценивает параметры конечных смесей распределений по выборке EM-подобными алгоритмами; компоненты смеси — параметрические распределения из `pysatl-core`.

## Карта раздела

- [Предметная область и сценарии](domain.md) — методы оценки и сценарий работы с библиотекой.
- [Архитектура](architecture.md) — модули и их зависимости.
- [Поведение](behavior.md) — что происходит при вызове `fit`.
- [Модули](modules/index.md) — устройство каждого модуля, его расширение и ограничения.

## Контекст системы

Лист System context показывает место библиотеки в окружении, форматы входных и выходных данных, а также внешние зависимости.

[Исходник схемы контекста](diagrams/full-architecture.drawio "System context")

- Вход: выборка `X` и начальная `MixtureModel`, которую пользователь сам собирает из компонент `pysatl-core` (`ParametricFamilyDistribution`).
- Настройка: пользователь выбирает оценщик (`EM`, `ECM`, `ECME`), критерии остановки (`breakpointers`), правила удаления компонент (`pruners`) и оптимизатор.
- Выход: `fit` возвращает обученную `MixtureModel`; историю итераций даёт свойство `history` (`IterationsHistory`).
- На листе на входе написан `EM(...)`; `ECM` и `ECME` принимают ту же базовую конфигурацию и дополнительно поддерживают разбиение параметров на группы.
- Два серых пути на листе запланированы: «`X` + семейства без параметров» (начальные параметры подбирает модуль `initializers`) и «только `X`» (модуль `structure_selection` подбирает число компонент и их семейства, затем начальные параметры) — Запланировано.
- Из зависимостей берутся:
  - `pysatl-core` — параметрические семейства и распределения (компонентой может быть любое из них), базовый класс `SamplingStrategy` для собственной стратегии `DeterministicSampling` и конечная смесь `FiniteMixtureDistribution` только для экспорта через `to_core_mixture`;
  - `scipy` — `optimize.minimize` (методы Nelder–Mead и Powell в `optimizers`) и `special.logsumexp`/`softmax`;
  - `numpy` — массивы.

## Стек

- Python: `>=3.12,<3.14`
- `pysatl-core`: `0.1.0a0`
- `numpy`: `>=2.2.3,<3.0.0`
- `scipy`: `>=1.15.2,<2.0.0`
- Документация: Sphinx + MyST (тема `pydata-sphinx-theme`)
- Диаграммы: draw.io

## Статус возможностей

| Возможность | Модуль | Статус |
| --- | --- | --- |
| Смесь распределений: плотность, логарифм правдоподобия, генерация выборки | [`core`](modules/core/core.md) | Готово |
| Готовые оценщики `EM`, `ECM`, `ECME` | [`estimators`](modules/estimators/index.md) | Готово |
| Собственный итеративный конвейер из шагов (`Pipeline`, `ExpectationStep`, `WeightsStep`, `MaximizationStep`) | [`estimators.iterative`](modules/estimators/iterative/iterative.md) | Готово |
| Критерии остановки (`StepBreakpointer`, `LikelihoodBreakpointer`) | [`estimators.iterative`](modules/estimators/iterative/iterative.md) | Готово |
| Удаление компонент с малым весом (`PriorThresholdPruner`) | [`estimators.iterative`](modules/estimators/iterative/iterative.md) | Готово |
| Численные оптимизаторы (`ScipyNelderMead`, `ScipyPowell`) | [`optimizers`](modules/optimizers/optimizers.md) | Готово |
| Прямые (неитеративные) методы оценки | [`estimators.direct`](modules/estimators/direct/direct.md) | Запланировано |
| Подбор начального приближения | [`initializers`](modules/initializers/initializers.md) | Запланировано |
| Выбор структуры смеси (число компонент и семейства) | [`structure_selection`](modules/structure_selection/structure_selection.md) | Запланировано |
