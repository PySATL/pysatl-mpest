"""Module which provides many useful utils for improving code writing experience"""

import functools
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterator, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class IteratorWrapper(Generic[T, R], Iterator[R]):
    """Class which represents custom iterator wrapper"""

    def __init__(
        self,
        instance: T,
        next_function: Callable[[T, int], R],
    ) -> None:
        self._instance = instance
        self._next_function = next_function
        self._ind = -1

    def instance(self):
        """Instance getter"""
        return self._instance

    def __next__(self):
        self._ind += 1
        return self._next_function(self._instance, self._ind)


class ANamed(ABC):
    """Class which represents named objects"""

    # pylint: disable=too-few-public-methods

    @property
    @abstractmethod
    def name(self) -> str:
        """Name getter"""


class Factory(Generic[T]):
    """Class which represents factory pattern"""

    # pylint: disable=too-few-public-methods

    def __init__(self, cls: type[T], *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def construct(self):
        """Get new instance of given class"""
        return self.cls(*self.args, **self.kwargs)


class ObjectWrapper(Generic[T]):
    """Class which wraps object and used for future inheritance"""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        content: T,
    ) -> None:
        self._content = content

    @property
    def content(self):
        """Wrapped object getter"""
        return self._content


class Indexed(ObjectWrapper[T]):
    """Class which wraps object, adding index field"""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        ind: int,
        content: T,
    ) -> None:
        super().__init__(content)
        self._ind = ind

    @property
    def ind(self):
        """Index getter"""
        return self._ind


class ResultWrapper(ObjectWrapper[T]):
    """Class which wraps result and used for future inheritance"""

    # pylint: disable=too-few-public-methods

    @property
    def result(self):
        """Result getter"""
        return self.content


class ResultWithError(ResultWrapper[T | None]):
    """
    Class which wraps result object, adding error field
    - Cant contains None value
    - Raises contained error on result getting attempt if exist
    - Raises ValueError("Empty result") if both result and error is None
    """

    def __init__(
        self,
        result: T | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(result)
        self._error = error

    @property
    def result(self) -> T:
        """
        Overrides ObjectWrapper.content
        - Raises contained error if exist
        - Raises ValueError("Empty result") if both content and error is None
        """
        if self.error is not None:
            raise self.error
        if self._content is None:
            raise ValueError("Empty result")
        return self._content

    @property
    def error(self):
        """Error getter"""
        return self._error


class ResultWithLog(Generic[T, R], ResultWrapper[T]):
    """Class which wraps result object, adding custom log field"""

    # pylint: disable=too-few-public-methods

    def __init__(self, result: T, log: R) -> None:
        super().__init__(result)
        self._log = log

    @property
    def log(self):
        """Log getter"""
        return self._log


class TimerResultWrapper(ResultWrapper[T]):
    """Class which wraps result object, adding time field"""

    # pylint: disable=too-few-public-methods

    def __init__(self, result: T, runtime: float) -> None:
        super().__init__(result)
        self._runtime = runtime

    @property
    def runtime(self):
        """Runtime getter"""
        return self._runtime


def apply(mapper: Callable[[R], T]):
    """Decorator which applies given map function to wrapped one"""

    def current_apply(func: Callable[P, R]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper_apply(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)
            return mapper(result)

        return wrapper_apply

    return current_apply


def timer(func: Callable[P, R]) -> Callable[P, TimerResultWrapper[R]]:
    """
    Decorator which replaces function output by TimerResultWrapper object
    and calculates runtime
    """

    @functools.wraps(func)
    def wrapper_timer(*args: P.args, **kwargs: P.kwargs) -> TimerResultWrapper[R]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        finish = time.perf_counter()
        runtime = finish - start
        return TimerResultWrapper(result, runtime)

    return wrapper_timer


def history(holder: list[T], mapper: Callable[[R], T] = lambda x: x):
    """
    Decorator factory which allows you
    to save custom mapped history of wrapped function calls to given list
    """

    def current_history(func: Callable[P, R]) -> Callable[P, R]:
        """
        Decorator which save custom mapped history of wrapped function calls to given list
        """

        @functools.wraps(func)
        def wrapped_history(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            holder.append(mapper(result))
            return result

        return wrapped_history

    return current_history


def logged(
    holder: list[TimerResultWrapper[T] | ObjectWrapper[T] | float],
    save_results: bool = True,
    save_results_mapper: Callable[[R], T] = lambda x: x,
    save_time: bool = True,
):
    """Decorator factory which simplifies using timer and history decorators"""

    def time_only(x: TimerResultWrapper[R]):
        return x.runtime

    def apply_mapper_to_result(x: ObjectWrapper[R]):
        return ObjectWrapper(save_results_mapper(x.content))

    def apply_mapper_to_timer_result(x: TimerResultWrapper[R]):
        return TimerResultWrapper(save_results_mapper(x.content), x.runtime)

    def current_logged(func: Callable[P, R]) -> Callable[P, ObjectWrapper[R]]:
        """Decorator which simplifies using timer and history decorators"""

        curr_func = apply(ObjectWrapper[R])(func)

        if save_time:
            if not save_results:
                history_decorator = history(holder, mapper=time_only)
            else:
                history_decorator = history(holder, mapper=apply_mapper_to_timer_result)
            curr_func = history_decorator(timer(func))
            return curr_func

        if save_results:
            return history(holder, mapper=apply_mapper_to_result)(curr_func)

        return curr_func

    return current_logged


def in_bounds(min_value: float, max_value: float):
    """Decorator for float result functions to set bonds of it's result values"""

    def current_in_bounds(func: Callable[P, float]) -> Callable[P, float]:
        @functools.wraps(func)
        def wrapper_apply(*args: P.args, **kwargs: P.kwargs) -> float:
            return min(max(func(*args, **kwargs), min_value), max_value)

        return wrapper_apply

    return current_in_bounds


def find_file(name, path):
    """
    A function for finding the path to a file
    """
    # pylint: disable=unused-variable
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

    return ""
