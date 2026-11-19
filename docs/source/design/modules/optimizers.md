# Optimizers


```{image} ../_static/optimizers.png
:alt: Select Parameters
:width: 600px
:align: center
```



## Классы

### Optimizer

Абстрактный класс для оптимизации параметров.

- **Атрибуты**

- **Методы**

  - ***\+ minimize(target: Callable, params: list[float]): list[float]***

    Абстрактный метод, который возвращает параметры, минимизирующие функцию `target`




Доступные оптимизаторы на данный момент:

1. `ScipyPowell` scipy реализация метода Powell
2. `ScipyNelderMead` scipy реализация метода NelderMead



## Различные диаграммы
