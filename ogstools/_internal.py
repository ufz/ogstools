# source: https://github.com/python/typing/issues/270#issuecomment-1344537820

import functools
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

S = TypeVar("S")
P = ParamSpec("P")
T = TypeVar("T")

# yet unused
# def copy_function_signature(
#     source: Callable[P, T]
# ) -> Callable[[Callable], Callable[P, T]]:
#     def wrapper(target: Callable) -> Callable[P, T]:
#         @functools.wraps(source)
#         def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
#             return target(*args, **kwargs)

#         return wrapped

#     return wrapper


def copy_method_signature(
    source: Callable[Concatenate[S, P], T],
) -> Callable[[Callable], Callable[Concatenate[S, P], T]]:
    def wrapper(target: Callable) -> Callable[Concatenate[S, P], T]:
        @functools.wraps(source)
        def wrapped(self: S, *args: P.args, **kwargs: P.kwargs) -> T:
            return target(self, *args, **kwargs)

        return wrapped

    return wrapper
