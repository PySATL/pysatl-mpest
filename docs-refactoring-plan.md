# План: docs-refactoring (перенос диаграмм на drawio)

Ветка: `iraedeus/docs-refactoring` (от `main`). Статус: **черновик, ждёт решений владельца**.

## Что нашли в текущей документации

Дизайн-доки описывают архитектуру до перехода на `pysatl-core`:

| Документ / диаграмма | Состояние |
|---|---|
| `modules/distributions.md`, `core.md` (`Parameter`, `ContinuousDistribution`) | Устарели: распределения теперь из `pysatl-core`; в `core/` только `MixtureModel`, `_components.py`, `_sampling.py` |
| `modules/strategies.md`, диаграмма `highlevel-module.png` (блок Strategies) | Устарели: стратегии стали приватными `estimators/iterative/_strategies/` (`q_function`, `observed_data_likelihood`, `moments`, `_dispatch`) |
| `estimators/iterative.md` | Частично: нет `ECM`/`ECME`/`EM` как готовых оценщиков, `_wave_planner`, `PipelineState`, `IterationsHistory`, `OptimizationBlock`, `WeightsStep` |
| `modules/initializers.md`, `components_number/` | Модулей в коде нет (initializers — следующая задача) |
| `estimators/direct.md` | Пакет `direct/` пуст |
| `Use-case.png` | Концептуально жив; «Add new distribution» и «Add new estimation strategy» надо пересмотреть |
| `activity-mpest.png`, `Algo-tree.png` | Концепт, не привязан к коду; Algo-tree по вашему мнению слабая |
| `docs/full-architecture.png`, `requirements_dashbard.html`, `usecases_dashboard.html` | Выяснить, нужны ли |

## Целевая раскладка

Работа идёт в новом проекте `docs_new/` (полноценный Sphinx-проект рядом с `docs/`); `docs/` и `docs.yml` до переключения не трогаем. Ниже пути относительно `docs_new/`.

```
source/design/
  diagrams/
    full-architecture.drawio        # 1 лист System context: вход/выход системы и зависимости
    module-architecture.drawio      # листы: Overview, Detailed
    usecases.drawio
    sequences.drawio                # листы seq-*: fit, E/M-шаг, pruning, ...
    activity.drawio
    algorithms-tree.drawio
  modules/<модуль>/
    <модуль>.md
    diagrams/<модуль>.drawio        # Overview, Packages, <подмодули...>, seq-*
```

Рядом с каждым `.drawio` лежит PNG-экспорт каждого листа (как в restaurant-staffing), md ссылается на PNG. Старые PNG из `_static/` удаляются после переноса.

Модули: `core`, `estimators` (+ `iterative`, `direct` — пустой), `optimizers`, `initializers` (пустой), `structure_selection` (пустой). Пустые модули рисуются стилем «planned».

## Этапы

1. **Инвентаризация и решения** — готово (этот план).
2. **Каркас**: `docs_new/` (копия `conf.py`, `Makefile`, `make.bat`, `_templates`, `api/`, `user_guide/`; `design/` пишется заново), каталоги, скрипт экспорта PNG из всех листов `.drawio` (`/opt/drawio/drawio`), пустые `.drawio` с листами, `docs/README.md` с правилами:
   - диаграммы: английские подписи, стиль planned-модулей, имена листов (`Overview`, `Packages`, `seq-*`), палитра, экспорт;
   - текст: стиль русского текста в md, что описывается в md, а что остаётся автогенерации `api/`.
3. **Верхний уровень** (диаграмма + текст вместе): `module-architecture`, `usecases`, `activity`, `algorithms-tree`, `sequences`, `full-architecture`; обновить `design/index.md` (стек, статусы).
4. **Модули по одному** (`core` → `optimizers` → `estimators`/`iterative`, затем planned: `initializers`, `structure_selection`, `direct`). Для каждого: сверка с кодом → диаграммы → md (назначение, ссылки на листы, классы по коду, ограничения). Первый вариант текста пишет Claude, владелец правит на ревью. Planned-модули — короткая страница «зачем и что планируется».
5. **Материалы**: раздел «Материалы» (math, review, дашборды) по решению 8.
6. **Переключение** (отдельный коммит): удалить `docs/`, переименовать `docs_new/` → `docs/`, перенести `math/`, `review/`, дашборды, поправить `.github/workflows/docs.yml` и ссылки (`README`, `CONTRIBUTING`), удалить старые PNG; сборка Sphinx без предупреждений.

Каждый этап — атомарные коммиты (`docs:`); проверка — сборка Sphinx. Работу ведёт Claude сам, без субагентов.

## Требования к диаграммам

- Диаграмма должна помещаться в страницу документации: ширина листа ≤ ~900 px, высота ≤ ~650 px, шрифт ≥ 11 pt. Слишком большая диаграмма делится на листы.
- Без пояснительного текста на диаграммах: заголовок, имена элементов, при необходимости подписи связей и легенда; пояснения — в md.
- Остальные правила — в `docs_new/README.md`.

## Решения владельца

1. **Модули.** `initializers`, `structure_selection` (бывш. `components_number`) и `estimators/direct` остаются как пустые (запланированные) модули; `distributions` и `strategies` удаляются. Остальные модули (`core`, `estimators/iterative`, `optimizers`) описываются по коду.
2. **Язык:** диаграммы на английском, md на русском (потом всё переводится).
3. **as-is по коду.** Модули, которых ещё нет в коде, выделяются отдельным стилем (planned).
4. `module-architecture`: два листа (Overview + Detailed).
5. **Экспорт PNG скриптом** (CLI `/opt/drawio/drawio`, тот же, что использует `drawio-diff`; скилл `drawio-skill` как справочник по XML).
6. Дашборды (`requirements_dashbard.html`, `usecases_dashboard.html`), блок `math/` и `review/` — готовятся к релизу; встроить в доку хотя бы ссылками, при необходимости вынести верхнеуровневую информацию.

7. **`full-architecture`:** один лист `System context` (id `system-context`), контекст системы. `pysatl-mpest` — один блок без модулей (модули — на `module-architecture`, пользователи — на `usecases`). Показаны входы от researcher (сейчас: `X` + начальная `MixtureModel`; planned: `X` + семейства без параметров через `initializers`, только `X` через `structure_selection`), выход (fitted `MixtureModel`, `IterationsHistory`) и что берётся из `pysatl-core`, `numpy`, `scipy`. Дополнительные листы (экосистема PySATL) — позже при необходимости.
8. **Дашборды, `math/`, `review/`:** новый раздел доки «Материалы»: по короткой странице на каждый вывод из `math/` (цель, ключевая формула, ссылка на PDF; соответствие стратегиям), краткие выводы из `review/` со ссылкой, дашборды как статические HTML (`html_extra_path`) со страницами-ссылками и аннотацией. Отдельная инфраструктура сбора статей и требований — позже (понадобится для конвейера реализации стратегий агентами).

9. **`algorithms-tree`:** таксономия предметной области: семейства методов (`EM-like`, `Monte Carlo`, `Direct`) → алгоритмы (`EM`, `ECM`, `ECME`), связи — обобщение. Без деталей реализации (шаги, стратегии, модули). `Monte Carlo` и `Direct` — planned.

## Остаётся обсудить

Открытых вопросов нет. Следующий шаг — этап 2 (каркас).
