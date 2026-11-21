# User Guide
**Назначение:** показать, как установить библиотеку, собрать начальную смесь, выбрать оценщик и прочитать результат оценки.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
mixtures
estimators
results
```

## Установка

Библиотека разрабатывается с [Poetry](https://python-poetry.org/) и пока устанавливается из репозитория:

```bash
git clone https://github.com/PySATL/pysatl-mpest.git
cd pysatl-mpest
poetry install
```

Нужен Python 3.12 или 3.13. Зависимости — `numpy`, `scipy` и `pysatl-core`, из которого берутся распределения компонент.
