# Модуль `initializers`
**Назначение:** описать проект модуля, который подбирает начальную смесь `MixtureModel` по выборке и заданным семействам компонент, и зафиксировать принятые решения (Запланировано).

```{toctree}
:hidden:
:maxdepth: 1

partitioners/partitioners
matchers/matchers
component_estimators/component_estimators
```

Модуль реализуется по этапам; ниже — проект, имена и сигнатуры предварительные. Пока модуля нет, начальную `MixtureModel` задаёт пользователь, см. [«Модуль `core`»](../core/core.md). Три роли `ClusterInitializer` разобраны на подстраницах: [«Разбиение»](partitioners/partitioners.md), [«Сопоставление кластеров и семейств»](matchers/matchers.md), [«Оценка параметров компоненты»](component_estimators/component_estimators.md).

## Задача

Инициализатор получает выборку `X` и список `families` (по элементу на компоненту) и возвращает `MixtureModel` с заданными параметрами и весами — старт для `fit` оценщиков смеси.

```python
normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

initializer = ClusterInitializer(KMeansPartitioner())

# Лучший из 10 запусков по правдоподобию стартовой смеси.
start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)

# Лучший из 10 запусков по итогу короткого EM, см. «Уточнение оценщиком смеси».
short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
refined = RefiningInitializer(initializer, short_em)
start = MultiStartInitializer(refined, n_init=10).initialize(X, [normal, normal, expon], random_state=42)

result = EM(...).fit(X, start)
```

Слово «оценщик» всегда уточняется:

- **оценщик смеси** — `EM`, `ECM`, `ECME` из `estimators`; поле `estimator` у `RefiningInitializer`;
- **оценщик компоненты** — `ComponentEstimator` в `initializers`, оценивает одну компоненту по кластеру; отсюда имена `component_estimator` и `ComponentEstimationError`.

Типы массивов — из `pysatl_mpest.typings`: `X` и столбец `h` — `UnivariateFloatArray`, матрица разбиения `H` — `MultivariateFloatArray`, перестановка `assignment` — `UnivariateIntArray`.

### Вход: `families`

- Элемент — `ParametricFamily` из `pysatl-core`. Фиксированные параметры и параметризация задаются через `family.view(...)`, результат (`PartialParametricFamily`) — тоже `ParametricFamily`. Без `view` — базовая параметризация.
- Оцениваются только свободные параметры. Компоненту собирает таблица через `family.distribution(**free)`, и её фиксированные параметры совпадают с `view`.
- Полностью известной компоненты (например, известного фона) в первой версии нет, см. «Отложено».
- Своего типа спецификации компоненты нет. Он понадобится с настройками отдельной компоненты: подсказками, границами параметров, своим способом оценки для одного из одинаковых семейств.

(init-output-order)=
### Выход: порядок компонент

- **Порядок компонент совпадает с `families`.** На него опираются настройки по номеру компоненты: `parameter_groups` у `ECM`, стратегии у `ECME`. Что компонента `j` — из `families[j]`, проверяет общий `Initializer.initialize`, см. {ref}`«Ход initialize» <init-walk>`.
- **Одинаковые семейства упорядочены по среднему кластера:** у `[normal, normal, normal]` компонента 0 — самая левая мода в любом запуске. Порядок наводит хук `_order` базового `Matcher`; исключение — `IdentityMatcher`, у него порядок задаёт способ разбиения, см. {ref}`«Порядок одинаковых семейств» <init-order-identical>`.

### Зависимости

- `core`, `pysatl-core` и `scipy`; k-means — `scipy.cluster.vq.kmeans2`, своего нет.
- scikit-learn — необязательная зависимость, только для `SklearnPartitioner`: `pip install pysatl-mpest[sklearn]`. Без него остальной модуль работает. В dev-зависимостях он есть, чтобы тесты `SklearnPartitioner` шли всегда.
- От `estimators` зависит только `RefiningInitializer`, и только от `BaseEstimator`. `estimators` от `initializers` не зависит, от `optimizers` модуль не зависит.
- Будущий `structure_selection` будет выдавать `families`: «`structure_selection` → `initializers` → `estimators`».

### Граница с `estimators`

Оценка параметров при известных ответственностях — по сути M-шаг, поэтому граница проведена по цели:

- **инициализатор даёт дешёвый старт:** явные формулы и одномерный поиск корня на отрезке, без итераций по смеси и без численной оптимизации, см. {ref}`«Контракт формулы» <init-formula-contract>`;
- **точное уточнение всей смеси — работа `estimators`;** исключение — семейства с носителем, зависящим от параметров, см. «Ограничения»;
- **`RefiningInitializer` границу не нарушает:** сам не итерирует, а вызывает готовый оценщик смеси, чтобы мультистарт сравнивал запуски по его итогу.

Нового по сравнению с EM в инициализаторе две вещи: получить ответственности без параметров и сопоставить кластеры с семействами.

## Состав модуля

В корне — публичный вход и готовые инициализаторы. Части схемы «разбиение → сопоставление → оценка» лежат в `cluster/`: у каждой роли свой подпакет с абстракцией в `base.py`. Каждый подпакет экспортирует свои публичные классы, пользователь импортирует всё из `pysatl_mpest.initializers`.

```text
initializers/
├── __init__.py
├── initializer.py
├── exceptions.py
├── cluster_initializer.py
├── multi_start.py
├── refining.py
└── cluster/
    ├── table.py
    ├── partitioners/
    │   ├── base.py
    │   └── ...
    ├── component_estimators/
    │   ├── base.py
    │   ├── _dispatch.py
    │   └── ...
    ├── matchers/
    │   ├── base.py
    │   ├── _matching.py
    │   └── ...
    └── _table/
        ├── _lazy.py
        ├── _pair.py
        └── _validation.py
```

| Модуль | Содержимое |
| --- | --- |
| `__init__.py` | Фасад: публичные классы и ошибки |
| `initializer.py` | Абстрактный `Initializer`: общий `initialize` с проверками входа и результата, абстрактный `_initialize` |
| `exceptions.py` | `InitializationError` (наследник `Exception`) и его наследник `ComponentEstimationError`, см. «Ошибки» |
| `cluster_initializer.py` | `ClusterInitializer`: «разбиение → сопоставление → оценка» |
| `multi_start.py` | `MultiStartInitializer`: мультистарт поверх любого `Initializer` |
| `refining.py` | `RefiningInitializer`: старт, доведённый оценщиком смеси |
| `cluster/partitioners/` | `Partitioner` и `KMeansPartitioner`, `SklearnPartitioner`, `QuantilePartitioner`, `RandomCentersPartitioner`, см. [«Разбиение»](partitioners/partitioners.md) |
| `cluster/component_estimators/` | `ComponentEstimator`, `MLEComponentEstimator`, позже `MomentsComponentEstimator` и `ComponentEstimatorChain`; `_dispatch.py` — диспетчер формул по ключу семейства, см. {ref}`«Реестр формул» <init-formula-registry>` |
| `cluster/matchers/` | `Matcher` с шаблонным `match`, `HungarianMatcher`, `IdentityMatcher`, позже `ExhaustiveMatcher`; `_matching.py` — чистые `solve_assignment(cost)` (см. {ref}`«Критерий по парам» <init-hungarian>`) и `order_identical(assignment, groups, means)` (см. {ref}`«Порядок одинаковых семейств» <init-order-identical>`) |
| `cluster/table.py` | Публичный протокол `ComponentTable` — всё, что видит сопоставитель, см. «Таблица компонент» |
| `cluster/_table/_lazy.py` | `LazyComponentTable` — единственная реализация протокола: ленивый кэш пар, группы одинаковых семейств, средние кластеров, сборка смеси |
| `cluster/_table/_pair.py` | Чистая `evaluate_pair`: оценка одной пары со всеми проверками, см. «Оценка пары» |
| `cluster/_table/_validation.py` | Чистые проверки: `validate_partition`, `validate_permutation`, предикаты порогов из «Вырожденных кластеров» |

В `core/_components.py` добавляются четыре функции. Им место в `core`: фиксированные параметры есть только у `PartialParametricFamily`, а он не экспортируется из `pysatl_core.families`, и все обращения к внутренностям `pysatl-core` живут там.

- `family_key(family)` — ключ «имя, параметризация, отсортированные фиксированные параметры»;
- `family_fixed_parameters(family)` — фиксированные параметры; `fixed_parameters(component)` становится её частным случаем;
- `family_free_parameter_names(family)` — имена свободных параметров, по ним `evaluate_pair` сверяет ответ оценщика;
- `component_family_key(component)` — `family_key` семейства компоненты, по нему общий `initialize` проверяет порядок результата.

Ещё в приватных модулях `core` — общие с `estimators` функция поиска точек выборки, не покрытых смесью ([#53](https://github.com/PySATL/pysatl-mpest/issues/53)), и константа `NUMERICAL_TOLERANCE` порога M-шага ([#47](https://github.com/PySATL/pysatl-mpest/issues/47)).

(init-first-version)=
### Первая версия

В `pysatl-core` 0.1.0a0 три семейства — `Normal`, `ContinuousUniform`, `Exponential`, — и у всех MLE в явном виде, у `Normal` и `Exponential` совпадающий с методом моментов. На них нельзя проверить без дублей, что дают `MomentsComponentEstimator`, `ComponentEstimatorChain` и `ExhaustiveMatcher`, поэтому они отложены.

| Часть | Первая версия | Позже |
| --- | --- | --- |
| Инициализаторы и таблица | `ClusterInitializer`, `MultiStartInitializer`, `RefiningInitializer`, `ComponentTable`, `LazyComponentTable` | — |
| Способы разбиения | `KMeansPartitioner`, `QuantilePartitioner`, `RandomCentersPartitioner`, `SklearnPartitioner` | — |
| Оценщики компоненты | `MLEComponentEstimator` с формулами для трёх семейств | `MomentsComponentEstimator` и `ComponentEstimatorChain` — с первым семейством, у которого MLE не сводится к явным формулам и одному корню, а моменты сводятся (например, GEV). `Gamma` и `Weibull` хватит `MLEComponentEstimator`. Закрывает INIT-05 |
| Сопоставители | `HungarianMatcher`, `IdentityMatcher` | `ExhaustiveMatcher` — с первым примером, где `HungarianMatcher` выбирает плохую раздачу |

Отложенные классы остаются на диаграмме, чтобы было видно, куда они встанут.

## Диаграмма классов

[Исходник диаграммы классов](diagrams/initializers.drawio "Overview")

- **`Initializer`** — минимальный интерфейс: `@final` `initialize(X, families, random_state=None)` приводит `X`, проверяет вход, создаёт генератор, вызывает абстрактный `_initialize(X, families, rng)` и проверяет результат. Кластерная схема — одна из реализаций.
- **`ClusterInitializer`** — единственный кластерный инициализатор с тремя заменяемыми ролями. Фасадов под способы разбиения нет: другой способ — другой `Partitioner`.
- **`MultiStartInitializer` и `RefiningInitializer`** — обёртки над любым `Initializer`. `RefiningInitializer` доводит старт оценщиком смеси `BaseEstimator`; это единственная связь с `estimators` (пунктир на рисунке).
- **Роли — агрегация:** `ClusterInitializer` получает их в конструкторе и не владеет ими, один объект роли можно отдать нескольким инициализаторам. `ComponentEstimatorChain` сам `ComponentEstimator` и содержит список оценщиков.
- **`Matcher`** устроен как `Initializer`: `@final` `match(table)` вызывает `_match(table)` и хук `_order`. `IdentityMatcher` переопределяет только хук.
- **`ComponentTable`** — протокол (кружок у `LazyComponentTable`). `Matcher` зависит только от него и не видит ни `X`, ни `H`, ни оценщика компоненты.
- **`LazyComponentTable`** создаётся на каждый вызов `initialize` и хранит оценщик компоненты в приватном поле; вызывает оценщик только она.

| Роль `ClusterInitializer` | По умолчанию |
| --- | --- |
| `Partitioner` | Нет, передаётся явно: способ разбиения сильнее всего влияет на результат |
| `ComponentEstimator` | `MLEComponentEstimator()`; с появлением цепочки — `ComponentEstimatorChain([MLEComponentEstimator(), MomentsComponentEstimator()])` |
| `Matcher` | `HungarianMatcher` |

На рисунке сознательно нет: зависимости `Matcher` от `ComponentTable` (видна из сигнатуры), приватных модулей функций (`_matching`, `_pair`, `_dispatch`, `_validation`), обращений к `core._components`, сборки `family.distribution(**free)` в `_pair` и вызовов `kmeans2` и `linear_sum_assignment` (последний — на схеме сопоставления).

## Устройство

(init-walk)=
### Ход `initialize`

[Исходник схемы вызовов](diagrams/initializers.drawio "Sequence: initialize")

`Initializer.initialize(X, families, random_state)`, общий и `@final`:

1. Приводит `X` к `UnivariateFloatArray` и проверяет: `families` не пуст, `X` одномерный, значения конечны, `n >= k`. Иначе `ValueError`.
2. Создаёт `rng = np.random.default_rng(random_state)`, см. «Случайность».
3. Вызывает `self._initialize(X, families, rng)`.
4. Проверяет результат (см. ниже) и возвращает смесь.

`ClusterInitializer._initialize(X, families, rng)`, где `k = len(families)`:

1. **Проверка входа** (см. ниже).
2. **Разбиение.** `Partitioner.partition(X, k, rng)` возвращает `H` размера `(n, k)`, столбец — кластер. `validate_partition(H, k)` проверяет структуру и приводит `H` к `MultivariateFloatArray`, затем проверяется размер кластеров, см. «Вырожденные кластеры». Порядок столбцов не меняется.
3. **Таблица.** Создаётся пустая `LazyComponentTable(X, H, families, component_estimator)`.
4. **Сопоставление.** `Matcher.match(table)` возвращает перестановку `assignment`: `families[j]` получает столбец `H[:, assignment[j]]`, одинаковые семейства уже упорядочены. `validate_permutation(assignment, k)` проверяет, что это перестановка.
5. **Сборка.** `table.mixture(assignment)` берёт компоненты из таблицы, считает веса и собирает `MixtureModel`. Пары, которые сопоставитель уже запрашивал, повторно не оцениваются; остальные (с `IdentityMatcher` или с `HungarianMatcher` при одинаковых семействах) оцениваются здесь.

[Исходник схемы сборки смеси](diagrams/initializers.drawio "Sequence: assembly")

**Проверка входа.** До разбиения, самой дорогой части, `ClusterInitializer` проверяет: `n >= k · min_cluster_size`; различных значений в выборке не меньше `k`; `component_estimator.supports(family)` для каждого семейства. Иначе `ValueError`, для неподдерживаемых семейств — `NotImplementedError` со списком всех таких семейств.

**Проверка перестановки.** `assignment` — перестановка `0…k−1`: длина `k`, целые, различные, в диапазоне. Иначе `ValueError` с полученным значением, например «Matcher вернул не перестановку длины 3: [0, 0, 1]», а не `IndexError` где-то в таблице.

**Проверка результата** — в общем `initialize`, поэтому гарантия «с этой смесью EM стартует» действует для любого инициализатора, в том числе пользовательского:

- в смеси `k` компонент, и `component_family_key(component) == family_key(families[j])`. Иначе `ValueError`: это ошибка в коде, мультистарт её не глотает;
- `loglikelihood(X)` конечен. Иначе `InitializationError` с примерами непокрытых точек; их находит та же функция из `core`, что проверяет пользовательский старт в `Pipeline.fit` ([#53](https://github.com/PySATL/pysatl-mpest/issues/53)). Другое разбиение может пройти, поэтому мультистарт такой запуск пропускает и сравнивает только конечные значения.

У `ClusterInitializer` бесконечное правдоподобие бывает, только когда раздачу выбрали без $\ell_{jc}$: с `IdentityMatcher` и с `HungarianMatcher` при одинаковых семействах. Пример: у `[expon, expon]` кластер `{−1, 5}` оценивается (`lambda_ = 0.5`), но точку −1 не покрывает ни одна компонента. Если её покрывает соседняя компонента, смесь годна: EM отдаст точку ей.

**Состояния нет.** Между стадиями передаются только `H`, таблица и `assignment`; таблица меняется только как кэш, и её ответы не зависят от порядка запросов. Порядок стадий фиксирован, поэтому роли передаются по имени, а не списком шагов, как в `Pipeline`.

### Роли

Действия 2 и 4 выполняют заменяемые роли; оценщик компоненты вызывается только из таблицы.

| Роль | Что делает | Подробно |
| --- | --- | --- |
| `Partitioner` | Разбивает выборку на `k` кластеров, возвращает `H` | [«Разбиение»](partitioners/partitioners.md) |
| `Matcher` | Решает, какой кластер достаётся какому семейству | [«Сопоставление кластеров и семейств»](matchers/matchers.md) |
| `ComponentEstimator` | Оценивает свободные параметры компоненты по кластеру | [«Оценка параметров компоненты»](component_estimators/component_estimators.md) |

(init-component-table)=
### Таблица компонент

`ComponentTable` — протокол: всё, что сопоставитель может спросить у таблицы. Пользовательские сопоставители пользуются только им, юнит-тесты подставляют поддельную таблицу.

| Член | Что возвращает |
| --- | --- |
| `k` (свойство) | Число семейств, оно же число кластеров |
| `groups()` | Группы одинаковых семейств, `tuple[tuple[int, ...], ...]`, см. «Одинаковые семейства» |
| `cluster_means()` | Средние кластеров по столбцам `H` |
| `loglikelihood(j, c)` | $\ell_{jc}$ — логарифм правдоподобия кластера $c$ при компоненте семейства `families[j]`, оценённой на нём; конечное число или $-\infty$ |
| `failure(j, c)` | Причина, по которой $\ell_{jc}$ не конечен, или `None` |

- **Протокол можно расширять:** таблицу создаёт только `ClusterInitializer`, пользователи пишут сопоставители, а не таблицы. Так `ExhaustiveMatcher` добавит `mixture_loglikelihood(assignment)`.
- **`component(j, c)` и `mixture(assignment)` — методы `LazyComponentTable`, а не протокола:** их зовёт только `ClusterInitializer`, сопоставителю собирать компоненты незачем.

`LazyComponentTable`:

- **ленивая:** пару оценивает при первом запросе через `evaluate_pair` и запоминает; одинаковые семейства делят записи, поэтому оценок не больше $d \cdot k$ при $d$ различных семействах, а не $k^2$;
- **нормализует значения:** `NaN` и $+\infty$ в `loglikelihood` записываются как $-\infty$;
- **собирает смесь:** `mixture(assignment)` — единственное место, где считаются веса и создаётся `MixtureModel`; выбранная пара без компоненты прерывает сборку `ComponentEstimationError`.

(init-pair-states)=
#### Состояния пары

| Состояние | Компонента | $\ell_{jc}$ | `failure` | `component` |
| --- | --- | --- | --- | --- |
| Оценена | есть | конечен | `None` | компонента |
| Не покрывает кластер | есть | $-\infty$ (в том числе из `NaN` и $+\infty$) | «1 точка кластера вне носителя, например −1.0» | компонента |
| Не оценена | нет | $-\infty$ | причина проверки | новый `ComponentEstimationError` с причиной |

- Пример «не покрывает»: `Exponential(0.5)` по кластеру `{−1, 5}`. Пример «не оценена»: `Exponential` на кластере со средним −2 даёт `lambda_ < 0`.
- `failure(j, c)` — `None` тогда и только тогда, когда $\ell_{jc}$ конечен.
- У «не оценена» `component` каждый раз бросает новое исключение, не вызывая оценщик повторно. Объект исключения не хранится: при повторном `raise` он копил бы кадры в `__traceback__` и держал ссылки на `X` и `H`.
- Причина хранится без номеров; префикс «семейство 2 (`Exponential`), кластер 1:» дописывается при чтении, потому что одну запись делят одинаковые семейства с разными номерами.

`HungarianMatcher` запрещает оба последних состояния, `IdentityMatcher` пропускает «не покрывает кластер» и оставляет решение проверке результата, см. {ref}`«Невозможные пары» <init-impossible-pairs>`.

(init-pair-walk)=
#### Оценка пары

[Исходник схемы оценки пары](diagrams/initializers.drawio "Sequence: estimate")

Пару «семейство `j`, кластер `c`» оценивает чистая `_pair.evaluate_pair(X, h, family, component_estimator, scale)`, где `h = H[:, c]`, `scale` = $\operatorname{std}(X)$ (таблица считает его один раз). О кэше и номерах она не знает, поэтому тестируется на литералах, пороги — ровно на границе. Возвращает `PairEvaluation(component, loglikelihood, reason)`, `component` — `None` у пары «не оценена».

1. `component_estimator.estimate(X, h, family)` возвращает словарь свободных параметров `free`. `ComponentEstimationError` делает пару не оценённой; `NotImplementedError` и ошибки в коде проходят насквозь.
2. **Ключи:** `set(free)` совпадает с `_components.family_free_parameter_names(family)`, иначе `TypeError` (ошибка в оценщике, например возвращён фиксированный параметр).
3. **Достаточность данных:** различных значений в кластере не меньше, чем свободных параметров, см. «Достаточность данных».
4. **Значения:** каждое — `int`, `float`, `np.integer` или `np.floating`, иначе `TypeError` (в том числе массив из одного элемента); приводится к `float`, чтобы не попал `np.float32`; неконечное делает пару не оценённой. Это нужно потому, что `pysatl-core` проверяет только ограничения параметризации и создаёт `Normal(mu="a")`, `Normal(mu=nan)`, `Normal(mu=np.array([1.0, 2.0]))` ([pysatl-core#158](https://github.com/PySATL/pysatl-core/issues/158)).
5. **Сборка** `family.distribution(**free)`. После шага 2 `ValueError` здесь означает только нарушенное ограничение параметризации (`lambda_ < 0`) и делает пару не оценённой.
6. **Ширина компоненты** по квартилям, см. «Численная вырожденность».
7. **Временно — пороги M-шага** у `Normal` и `Exponential`, см. там же.
8. **$\ell_{jc}$** по точкам кластера (`h > 0`). Не конечен — пара не покрывает кластер: компонента сохраняется, причина называет точки вне носителя.

Шаги 3–7 переводят пару в «не оценена» с причиной этой проверки. Проверки одинаковы для любого оценщика, поэтому свой оценщик забыть их не может.

**У пары «не покрывает кластер» параметры — не MLE:** с точкой вне носителя правдоподобие кластера равно нулю при любых параметрах. Формула учитывает такую точку с весом 1: у `Exponential` по `{−1, 5}` `lambda_ = 1/2` вместо `1/5`. EM исправляет это на первой итерации: E-шаг даёт точке ответственность 0, M-шаг пересчитывает параметр без неё. Оценка только по точкам внутри носителя — см. «Отложено».

(init-identical-families)=
#### Одинаковые семейства

- Ключ семейства — «имя, параметризация, фиксированные параметры», `_components.family_key(family)`. `Normal` с `mu=0` и с `mu=5` — разные семейства.
- Группы считает только таблица и отдаёт `groups()` — кортежами индексов в порядке первого вхождения; кортежи неизменяемы, сопоставитель не испортит кэш.
- Свои ключи никто больше не считает: при расхождении `normal.view(mu=0)` мог бы получить из общей записи компоненту свободного `normal`.

(init-randomness)=
### Случайность

Случайность — аргумент вызова, а не поле объекта. `initialize` принимает `random_state` — `int`, `np.random.Generator` или `None`, как `MixtureModel.generate`. Генератор `rng` создаёт общий `initialize`, `ClusterInitializer` передаёт его в `partition(X, k, rng)`. Своего генератора нет ни у одной роли.

| `random_state` | Поведение |
| --- | --- |
| Целое число | Одинаковые аргументы — одинаковая смесь, сколько бы раз объект ни вызывали раньше |
| `np.random.Generator` | Используется на месте и сдвигается, как в `generate`; два вызова с одним генератором дают разные смеси |
| `None` | Свежая энтропия, не воспроизводится |

- Воспроизводимость с целым сидом нужна `structure_selection`: каждому числу компонент — свой сид или потомок через `spawn`.
- Детерминированный способ разбиения (`QuantilePartitioner`) `rng` не использует; `SklearnPartitioner` переводит его в целый сид, см. {ref}`SklearnPartitioner <init-sklearn-partitioner>`.

### Веса

Вес компоненты $j$ — средняя ответственность её кластера, считает `table.mixture(assignment)`; отдельной роли нет:

$$
\pi_j = \frac{1}{n} \sum_{i=1}^{n} \gamma_{i\sigma(j)}.
$$

(init-degenerate)=
### Вырожденные кластеры

Вырожденный кластер — пустой, со слишком малым числом точек или без разброса (одинаковые или почти одинаковые значения). Проверки распределены так:

| Что проверяется | Кто и когда | Ошибка |
| --- | --- | --- |
| Форма `H` — `(n, k)`, значения 0 и 1, в строке ровно одна единица | `validate_partition` после разбиения | `ValueError` |
| Пустой кластер | `KMeansPartitioner` при разбиении, иначе `validate_partition` | `InitializationError` |
| Размер кластера не меньше `min_cluster_size` | `ClusterInitializer` после разбиения | `InitializationError` |
| Различных значений не меньше, чем свободных параметров | `evaluate_pair` | `ComponentEstimationError` |
| Параметры конечны и допустимы (`NaN`, `sigma = 0`, `lambda_ < 0`) | `evaluate_pair` | `ComponentEstimationError` |
| Ширина компоненты больше порога | `evaluate_pair` | `ComponentEstimationError` |
| Дисперсия `Normal` и среднее `Exponential` больше `NUMERICAL_TOLERANCE` | `evaluate_pair`, временно | `ComponentEstimationError` |

Все пороги — чистые предикаты в `cluster/_table/_validation.py`, тестируемые ровно на границе.

#### Структура разбиения

- Способ разбиения и `ClusterInitializer` проверяют только структуру и кластеризаторы не чинят: опустевшие центры переносит только `KMeans` из scikit-learn.
- Неверная структура `H` — `ValueError` (ошибка в способе разбиения), пустой столбец — `InitializationError`.
- «Шумовых» меток (`-1` у `DBSCAN`) и неверного числа кластеров у `SklearnPartitioner` не бывает: он принимает только кластеризаторы с `n_clusters`.
- Ошибки входа — `ValueError` до разбиения: `n < k` проверяет общий `initialize`, `n < k · min_cluster_size` и «различных значений меньше `k`» — `ClusterInitializer`.

#### Достаточность данных

- **Нижнюю границу проверяет `evaluate_pair`** для любого оценщика: различных значений в кластере не меньше, чем свободных параметров, иначе условий меньше, чем неизвестных. `Normal` нужны два различных значения.
- **Сверх неё требования ставит оценщик компоненты:** только он знает, сколько данных нужно его формулам.
- **Ошибка понятная:** недостаток данных и недопустимые параметры (`ValueError` из `pysatl-core`) становятся `ComponentEstimationError` с причиной, и пара становится невозможной, см. {ref}`«Невозможные пары» <init-impossible-pairs>`.
- **Цепочка не передаёт кластер дальше:** `ComponentEstimationError` в `ComponentEstimatorChain` не отдаёт кластер следующему способу, иначе одинаковые семейства оценивались бы разными способами.

#### Численная вырожденность

Правдоподобие смеси не ограничено сверху: компонента на группе почти совпадающих точек с масштабом, стремящимся к нулю, даёт сколь угодно большое правдоподобие (Kiefer и Wolfowitz, 1956; McLachlan и Peel, 2000). Такой «максимум» смысла не имеет, и масштаб компоненты принято ограничивать снизу: так делает ограниченная оценка максимального правдоподобия у Hathaway (1985) и `reg_covar = 1e-6` в `GaussianMixture` из scikit-learn. Ограничение `sigma > 0` в `pysatl-core` не помогает: ровно ноль в арифметике с плавающей точкой обычно не получается.

Инициализатору ограничение нужнее, чем EM: он сам выбирает старт по правдоподобию (`HungarianMatcher` — раздачу, `MultiStartInitializer` — запуск), и вырожденный старт выигрывает любое сравнение, а EM из него не уходит. Поэтому `evaluate_pair` требует, чтобы ширина компоненты — расстояние между квартилями — была не слишком мала по сравнению с выборкой:

$$
\operatorname{ppf}(0.75) - \operatorname{ppf}(0.25) > \varepsilon \cdot \operatorname{std}(X), \qquad \varepsilon = 10^{-6}.
$$

- **Срабатывает только на атомах** — многих почти совпадающих значениях: залипший датчик, заглушка вместо пропуска, цензурирование на пороге, округление с пересчётом единиц. Ровно одинаковые значения раньше отвергает достаточность данных; на непрерывных данных без атомов проверка ничего не меняет.
- **Одна формула для всех семейств:** квартили даёт `calculate_characteristic(PPF, ...)`, знать параметр масштаба не нужно. Ширина `Normal(0, 2)` — 2.70, `Exponential(0.5)` — 2.20, `ContinuousUniform(0, 4)` — 2.0.
- **Порог относительный** и не зависит от единиц. Справа $\operatorname{std}(X)$, а не размах квартилей выборки: он равен 0, если совпадает больше половины значений.
- **$\varepsilon$ — соглашение** порядка `reg_covar`, а не выведенная величина. Узкие кластеры из нескольких выбросов (`20.0` и `20.001`, `sigma = 0.0005`) порог пропускает, их отсекает `min_cluster_size`.
- **Временная проверка порогов M-шага.** M-шаг не обновляет `sigma` у `Normal` при дисперсии не больше `NUMERICAL_TOLERANCE = 1e-9` (`sigma` ≲ `3e-5`) и `lambda_` у `Exponential` при взвешенном среднем не больше `1e-9`, и такой старт застрял бы навсегда. Относительная ширина этого не ловит: при $\operatorname{std}(X) = 10^{-4}$ `sigma = 1e-5` её проходит. Поэтому такая пара не оценена с причиной «дисперсия (среднее) меньше абсолютного порога M-шага `1e-9`, перемасштабируйте данные». Константа общая с `estimators` и лежит в `core`. Проверка уйдёт, когда порог M-шага станет относительным ([#51](https://github.com/PySATL/pysatl-mpest/issues/51)), см. «Отложено».

````{admonition} Пример: залипший датчик
:class: tip

Две моды `N(0, 1000)` и `N(6000, 1500)` (2000 точек) и 40 показаний залипшего датчика `15000 ± 0.001`; `[normal, normal]`, мультистарт из 100 запусков `RandomCentersPartitioner`, затем `EM`. Один запуск выделяет залипший блок в кластер с `sigma ≈ 9e-4`, который пропускают все остальные проверки.

```{figure} diagrams/initializers.width-check-example.png
:alt: Четыре графика: данные, правдоподобие стартов мультистарта, итог EM без проверки ширины и с ней

Без проверки ширины запуск с пиком выигрывает по правдоподобию (2), и `EM` теряет обе моды (3); с проверкой он пропускается, и `EM` их находит (4). Оценка пар и выбор запуска повторены вручную по правилам этой страницы.
```
````

#### Порог размера кластера

Численный порог узкие кластеры из нескольких выбросов не ловит, а оба критерия выбора (`HungarianMatcher` и мультистарт) их предпочитают, поэтому порог размера включён по умолчанию:

- `ClusterInitializer(min_cluster_size=None)` требует не меньше $2p$ точек в кластере, $p$ — наибольшее число свободных параметров среди `families`: 4 для `Normal` и `ContinuousUniform`, 2 для `Exponential`;
- целое значение задаёт порог явно, `min_cluster_size=1` отключает проверку;
- кластер меньше порога — `InitializationError`, мультистарт такие разбиения отбрасывает;
- проверка — на действии 2, до сопоставления, и не зависит от семейства.

#### Чего нет

- **Молчаливых подстановок** вроде «оценки по всей выборке»: смесь выглядела бы нормально, но была бы построена не так, как думает пользователь.
- **Повторных попыток внутри:** это дело `MultiStartInitializer`.

(init-errors)=
### Ошибки

- **Неудачное разбиение** — `InitializationError` и его наследник `ComponentEstimationError`: другое разбиение может пройти, поэтому их ловят `evaluate_pair` (только `ComponentEstimationError` оценщика) и `MultiStartInitializer`. `InitializationError` наследует `Exception`, а не `ValueError`, чтобы `except ValueError` не ловил неудачные разбиения.
- **Всё остальное** (`ValueError`, `NotImplementedError`, `ImportError`, `TypeError`) пробрасывается: ошибки входа, конфигурации и кода другое разбиение не исправит.

| Ситуация | Ошибка | Где возникает |
| --- | --- | --- |
| `families` пуст, `X` не одномерный или с неконечными значениями, `n < k` | `ValueError` | Общий `Initializer.initialize` |
| `n < k · min_cluster_size`, различных значений меньше `k` | `ValueError` | Проверка входа `ClusterInitializer` |
| Семейство не поддерживается оценщиком компоненты | `NotImplementedError` со списком всех таких семейств | Проверка входа `ClusterInitializer` |
| scikit-learn не установлен | `ImportError` | Конструктор `SklearnPartitioner` |
| Кластеризатор без `n_clusters` | `ValueError` | Конструктор `SklearnPartitioner` |
| Неверная структура `H` | `ValueError` | `validate_partition` |
| Пустой кластер | `InitializationError` | `KMeansPartitioner` или `validate_partition` |
| Кластер меньше `min_cluster_size` | `InitializationError` | `ClusterInitializer` после разбиения |
| Пару «семейство — кластер» нельзя оценить | `ComponentEstimationError` | Оценщик компоненты или `evaluate_pair`; на сопоставлении пара невозможна |
| Оценщик вернул не те параметры или не вещественный скаляр | `TypeError` | `evaluate_pair`; ошибка в оценщике или формуле |
| В матрице стоимостей `NaN` или $-\infty$ | `ValueError` | `solve_assignment`; ошибка в коде |
| Нет раздачи, где каждое семейство получает подходящий кластер | `InitializationError` с причинами невозможных пар | `HungarianMatcher`, `ExhaustiveMatcher` |
| Пару, выбранную без $\ell_{jc}$ (`IdentityMatcher`, `HungarianMatcher` при одной группе), нельзя оценить | `ComponentEstimationError` | Сборка смеси |
| Сопоставитель вернул не перестановку | `ValueError` | `validate_permutation` после `match` |
| Вариантов раздачи больше `max_candidates` | `ValueError` | `ExhaustiveMatcher` |
| Смесь не покрывает часть выборки | `InitializationError` с примерами точек | Проверка результата в общем `initialize` |
| Не `k` компонент или не в порядке `families` | `ValueError` | Проверка результата в общем `initialize`; ошибка в коде |
| Все запуски мультистарта неудачны | `InitializationError`, причина последнего в `__cause__` | `MultiStartInitializer` |
| `fit_kwargs` не подходят к `estimator.fit` (неизвестное имя, повтор `X` или `mixture`) | `TypeError` | Конструктор `RefiningInitializer` |
| Оценщик смеси вернул меньше `k` компонент | `InitializationError` | `RefiningInitializer` после `estimator.fit` |

### Мультистарт

[Исходник схемы мультистарта](diagrams/initializers.drawio "Sequence: multistart")

`MultiStartInitializer(initializer, n_init)` вызывает `initializer.initialize(X, families, random_state=child)` `n_init` раз и возвращает смесь с наибольшим `loglikelihood(X)`.

- **Запуски независимы:** генератор делится через `rng.spawn(n_init)`, поэтому запуски различаются, воспроизводятся при целом сиде и могут идти параллельно.
- **Неудачные запуски пропускаются:** ловится только `InitializationError`. Если упали все — `InitializationError` с причиной последнего. Так мультистарт заодно закрывает редкие вырожденные разбиения.
- **Пропуски не молчат:** если упал хотя бы один, но не все, — `RuntimeWarning` «3 из 10 запусков не удались; последняя причина: …». `RuntimeWarning`, как у событий во время работы в `estimators`.
- **С детерминированным инициализатором** (`QuantilePartitioner`, `SklearnPartitioner(AgglomerativeClustering())`) мультистарт бесполезен.
- **Критерий — правдоподобие того, что вернул вложенный инициализатор:** у `ClusterInitializer` это старт, дёшево, но лучший старт не обязательно даёт лучший итог. Критерий не ограничен и тянется к узким кластерам, от них защищают проверки из «Вырожденных кластеров». Сравнение по итогу оценщика — через `RefiningInitializer`.

(init-refining)=
### Уточнение оценщиком смеси

[Исходник схемы уточнения](diagrams/initializers.drawio "Sequence: refining")

`RefiningInitializer(initializer, estimator, fit_kwargs=None)` — `Initializer`, у которого `_initialize` получает старт от `initializer` и возвращает `estimator.fit(X, start, **fit_kwargs)`. Мультистарт по итогу оценщика (INIT-02) — `MultiStartInitializer` вокруг него, без изменений в самом мультистарте:

- **emEM:** оценщик с коротким остановом (`StepBreakpointer(20)`), лучший кандидат потом доводится полным `EM`, как в примере из «Задачи»;
- **полный прогон на каждый старт:** лучший запуск — уже итоговая смесь.

Отдельного фасада «мультистарт + оценщик» нет: он повторил бы логику `MultiStartInitializer`, а его сигнатура не подошла бы ни к `Initializer`, ни к `BaseEstimator`. При необходимости его можно добавить функцией в одну строку.

- **`fit_kwargs`** уходят в `estimator.fit` как есть (`parameter_groups` у `ECM`, стратегии `ECME`). Конструктор копирует словарь и проверяет его через `inspect.signature(estimator.fit).bind(X, mixture, **fit_kwargs)` с заглушками: ошибка — `TypeError` при создании, а не на первом запуске. Номера компонент совпадают с номерами `families`, см. {ref}`«Выход: порядок компонент» <init-output-order>`.
- **Удаление компонент:** с pruners оценщик может вернуть меньше `k` компонент. Общая проверка сочла бы это ошибкой кода, поэтому `RefiningInitializer` сам бросает `InitializationError` «оценщик вернул 2 компоненты из 3», и мультистарт пропускает запуск.
- **Ошибки:** `InitializationError` вложенного инициализатора доходит до мультистарта как есть, ошибки оценщика смеси пробрасываются.
- **Случайность:** `rng` уходит только во вложенный инициализатор, оценщики смеси детерминированы.
- **Цена:** `n_init` запусков — `n_init` вызовов `fit`; запуски можно вести параллельно.
- **Критерий тянется к вырожденным решениям** сильнее, чем на старте, см. «Ограничения».

## Как расширить

- **Кластеризатор scikit-learn:** поставить `pysatl-mpest[sklearn]` и передать кластеризатор в `SklearnPartitioner`.
- **Способ разбиения:** унаследовать `Partitioner`, реализовать `partition` — матрица `(n, k)` с ровно одной единицей в строке, тип `bool`, `int` или `float`; случайность только из `rng`. Если способ знает семейства и кладёт кластер для `families[j]` в столбец `j`, использовать с `IdentityMatcher`.
- **Формулы для нового семейства `pysatl-core`** (доработка библиотеки): функция `formula(X, h, fixed) -> dict[str, float]`, зарегистрированная вызовом в приватном реестре способа для пары «семейство, параметризация», см. {ref}`«Контракт формулы» <init-formula-contract>`.
- **Способ оценки:** унаследовать `ComponentEstimator`, реализовать `supports` и `estimate`, вернуть словарь свободных параметров. Нехватку данных для своих формул — `ComponentEstimationError`, остальное проверит `evaluate_pair`. В цепочку по умолчанию — только способы в явном виде.
- **Сопоставитель:** унаследовать `Matcher`, реализовать `_match` — перестановку длины `k`, пользуясь только `ComponentTable`. Хук `_order` переопределять, только если порядок задаёт кто-то другой.
- **Инициализатор другого устройства:** унаследовать `Initializer` и реализовать `_initialize(X, families, rng)`; проверки входа и результата сделает общий `initialize`, мультистарт работает без изменений.

## Ограничения и решения

(init-limitations)=
### Ограничения

- **Носитель, зависящий от параметров, оценщик смеси не расширяет.** Старт `ContinuousUniform` — диапазон кластера, а в EM границы могут только сужаться (точки вне носителя получают ответственность 0); `ECME` расширяет их лишь частично. Пример: на 600 точках `U(0, 10)` и 400 точках `N(5, 0.5)` старт `U(0.05, 5.82)` после 100 итераций EM не меняется, у `ECME` становится `U(0.05, 5.94)`, нормальная компонента расплывается до `N(5.97, 1.84)`, логарифм правдоподобия −2166 против −2043 у истинной смеси. Нужен мультистарт или ручной старт.
- **Способ разбиения не знает семейств:** при разных носителях (`Normal` и `Exponential`) кластеры могут не подойти ни одному семейству; сопоставление выбирает лучшую пару, но разбиение не исправляет.
- **Способ разбиения не учитывает фиксированные параметры.** Обычно это исправляет сопоставление: у `[normal.view(mu=0), normal]` на модах около 0 и 10 семейство с `mu=0` получит кластер около 0. Но если ни один кластер не лежит около фиксированного значения, стартовая `sigma` будет смещена; исправляют EM или мультистарт.
- **Формулы регистрируются по параметризации.** У `Normal` их четыре; `pysatl-core` переводит значения в базовую параметризацию, но не семейство с фиксированными параметрами, поэтому формулы `meanStd` для `meanVar` не переиспользуются. `normal.view(parametrization_name="meanVar")` даст `NotImplementedError`, пока нет своих формул; потом `normal` и он будут разными семействами — смесь не портится, но оценок больше и порядок этих компонент не нормализован.
- **Мультистарт по итогу оценщика тянется к вырожденным решениям:** EM может сжать компоненту на паре близких точек, и такой запуск выиграет. После оценщика защищает только порог M-шага; своей защиты в `initializers` нет, это задача M-шага, см. «Отложено».
- **Сопоставление оптимально только относительно оценок выбранного способа,** а не в максимуме правдоподобия.
- **`ExhaustiveMatcher` растёт комбинаторно** и неприменим при многих разных семействах.
- **Семейства без явных формул и одного корня пока не инициализируются:** численный способ оценки отложен.
- **Агломеративное и спектральное разбиение квадратичны по $n$** и годятся до ~$10^4$ точек (`ward` при $n = 20\,000$ — 9 с и ~1,6 ГБ), см. [«Разбиение»](partitioners/partitioners.md).
- **Только жёсткое разбиение и одномерные выборки.**

### Принятые решения

Сводка; полные формулировки — в разделах.

- **Интерфейс:** `@final` `initialize` с общими проверками, реализации пишут `_initialize`; случайность — аргумент вызова («Ход `initialize`», «Случайность»).
- **Разбиение:** только жёсткое, способ передаётся явно; k-means на `kmeans2`, scikit-learn — через `SklearnPartitioner` с `n_clusters`; случайный старт — по центрам ([«Разбиение»](partitioners/partitioners.md)).
- **Сопоставление:** `HungarianMatcher` по умолчанию, `IdentityMatcher`; сопоставителя «слева направо» нет — это несколько строк в `_match`; одинаковые семейства — по среднему кластера ([«Сопоставление»](matchers/matchers.md)).
- **Таблица:** протокол `ComponentTable` и ленивая `LazyComponentTable`, пару проверяет `evaluate_pair`; три состояния пары («Таблица компонент»).
- **Вырожденные кластеры:** ширина по квартилям больше $10^{-6} \cdot \operatorname{std}(X)$, `min_cluster_size` = $2p$, без подстановок («Вырожденные кластеры»).
- **Ошибки и мультистарт:** неудачное разбиение — `InitializationError`, остальное пробрасывается; мультистарт над любым `Initializer`, по итогу оценщика — через `RefiningInitializer` («Ошибки», «Мультистарт»).
- **Оценка компоненты:** только `MLEComponentEstimator`, формулы — словарь свободных параметров, приватный реестр ([«Оценка параметров компоненты»](component_estimators/component_estimators.md)).

(init-deferred)=
### Отложено

- **Полностью известная компонента** (например, фон). Через `view` не выразить: `pysatl-core` запрещает фиксировать все параметры. Нужны заморозка целой компоненты в `estimators` (`EM` оптимизирует все свободные параметры, `ECM` не даёт оставить компоненту без групп), свой ключ и другой расчёт веса: жёсткий кластер занижает вес перекрытого фона.
- **Встроенный способ разбиения, знающий семейства,** например k-means с неподвижными центрами для фиксированных параметров положения. Нужно передать `families` в `partition`; сопоставитель для него, `IdentityMatcher`, уже есть.
- **Относительный порог M-шага** ([#51](https://github.com/PySATL/pysatl-mpest/issues/51)). Тогда временная проверка порогов M-шага уходит, а порог ширины согласуется с новым: при пороге $10^{-9} \cdot \operatorname{var}(X)$ нынешняя ширина гарантирует лишь $\sigma^2 \gtrsim 5.5 \cdot 10^{-13} \cdot \operatorname{var}(X)$.
- **Оценка по точкам внутри носителя** для семейств с носителем, не зависящим от параметров (`Exponential`). Нужен носитель семейства до оценки, а в `pysatl-core` он есть только у готового распределения. Пока это исправляет первая итерация EM.
- **Численный способ оценки через `Optimizer`** — для семейств без явных формул и одного корня. Нужна начальная точка, и численные оценки должны быть общими с `estimators`. Вместе с первым таким семейством.
- **Мягкое разбиение.** Нужно заново определить принадлежность к кластеру, размер кластера и `min_cluster_size`, носитель и невозможные пары, границы `ContinuousUniform`. Сигнатура `estimate(X, h, family)` останется, `h` станет весами.
- **Оценка по квантилям** (медианы и квартили, INIT-04) — устойчива к выбросам до четверти кластера. Для семейств «сдвиг и масштаб» (`Normal`, `ContinuousUniform`, `Exponential`, Gumbel, Лапласа) это одна общая функция от двух выборочных квантилей, например $\sigma = \mathrm{IQR} / 1.349$ у `Normal`; для логнормального, Вейбулла, Парето — то же по $\ln x$; для семейств с одним параметром формы (Gamma, GEV) форма находится `brentq` по отношению квантилей. Делать с первым случаем, где выбросы портят MLE.
- **Оценка по L-моментам** — устойчивее к выбросам, подходит для GEV, Gumbel, Pareto и закрывает тяжелохвостую часть INIT-05: L-моменты существуют, если есть среднее. Нужно решить, как работать с мягким разбиением.
- **Ускорение сопоставления подвыборкой**, см. {ref}`«Ускорение» <init-speedup>`.
- **Разбиение подвыборки** для `AgglomerativeClustering` и `SpectralClustering`: остальные точки — к ближайшему центру. Обёрткой над `Partitioner` или параметром `SklearnPartitioner` — решить.
- **Возмущённые старты** (INIT-02): инициализатор, случайно сдвигающий параметры готовой смеси, — когда понадобится.
- **Регуляризация дисперсии в M-шаге,** аналог `reg_covar`: ограничивает правдоподобие и защищает мультистарт по итогу оценщика ценой небольшого смещения. Изменение в `estimators`.
- **Подменяемая таблица компонент:** со второй реализацией `ClusterInitializer` получит `table_factory` (по умолчанию `LazyComponentTable`) и протокол `AssemblingTable(ComponentTable)` с `component` и `mixture`, без поломки существующего кода.
- **Общий диспетчер в `core`** ([#52](https://github.com/PySATL/pysatl-mpest/issues/52)) — после реализации модуля. `FamilyDispatch` берёт ключ у компоненты из первого аргумента и имеет реализацию по умолчанию, диспетчер формул берёт ключ у семейства и умолчания не имеет. Общим будет реестр «ключ → реализация», вызов по первому аргументу останется обёрткой в `estimators`. Общие с M-шагом формулы — позже, см. {ref}`«Реестр формул» <init-formula-registry>`.
