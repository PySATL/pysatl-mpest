# Модули
**Назначение:** показать, из каких модулей состоит библиотека, что делает каждый и где о нём читать.

```{toctree}
:hidden:
:maxdepth: 1

core/core
optimizers/optimizers
estimators/index
initializers/initializers
structure_selection/structure_selection
```

## Список модулей

| Модуль | Что делает | Статус |
| --- | --- | --- |
| [`core`](core/core.md) | Модель смеси `MixtureModel` поверх распределений `pysatl-core` | Готово |
| [`optimizers`](optimizers/optimizers.md) | Численные оптимизаторы для M-шага | Готово |
| [`estimators`](estimators/index.md) | Готовые оценщики `EM`, `ECM`, `ECME` | Готово |
| [`estimators.iterative`](estimators/iterative/iterative.md) | Итерационный конвейер: шаги, критерии остановки, удаление компонент, история | Готово |
| [`estimators.direct`](estimators/direct/direct.md) | Неитерационные оценщики | Запланировано |
| [`initializers`](initializers/initializers.md) | Подбор начальных параметров по выборке | Запланировано |
| [`structure_selection`](structure_selection/structure_selection.md) | Выбор числа компонент и их семейств | Запланировано |

Зависимости между модулями показаны на странице [«Архитектура»](../architecture.md). Читать удобно в порядке зависимостей: `core`, `optimizers`, `estimators`, `estimators.iterative`.

## Устройство страниц

Страница реализованного модуля состоит из одних и тех же разделов: состав модуля, диаграмма классов, устройство его механизмов, как расширить модуль и ограничения. Сигнатуры и параметры на страницах реализованных модулей не повторяются — они в [справочнике API](../../api/index.rst). Страница запланированного модуля содержит назначение, а когда модуль готовится к реализации — и его дизайн с сигнатурами, как у `initializers`: справочника API для ненаписанного кода ещё нет. После реализации сигнатуры уходят в справочник.
