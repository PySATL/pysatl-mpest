# Strategies

Стратегии обновления компонент смеси. Реализованы с помощью `@singledispatch`.

У каждого шага оценки параметров должна быть своя стратегия, например для оптимизации Q-функции внутри ComponentMaximization, в папке `pysatl-mpest/algorithms/strategies`:

```python
@singledispatch
def optimize_q_function(X: ndarray, W: ndarray, optimizer: ScipyOptimizer):
	...
```

Эта функцией будет как бы "родительской" для остальных. Если не реализована конкретная стратегия оптимизации Q-функции для конкретной компоненты, будет вызываться она. Теперь необходимо реализовать эту стратегию для каждой компоненты, например для нормального распределения, и зарегистрировать её:

```python
@optimize_q_function.register(Normal)
def optimize_q_function_normal(X: ndarray, W: ndarray, optimizer: ScipyOptimizer):
	...
```



## Классы

Их пока нет. Возможно придется потом сделать для того, чтобы делать fallback для некоторых компонент.



## Различные диаграммы
