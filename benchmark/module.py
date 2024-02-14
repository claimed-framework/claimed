"""Module functions."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

from dataclasses import dataclass
from typing import Callable


def hello_world() -> str:
    """Return Hello World."""
    return "Hello World"


class Foo:
    """An example class."""

    def __init__(self, a: int) -> None:
        """Initialize Foo.

        Args:
            a : documentation for argument a.
        """
        self.a = a

    def method_that_would_really_waste_your_time_if_it_fails(self) -> str:
        """Static typing could help you fix a bug in here before running any test.

        Returns:
            documentation for the returned string
        """
        self.a_times_1 = [1] * self.a
        # example that would trigger a mypy typechecking failure
        # return self.a + "When will you find out that this fails?"
        return f"{self.a} This works"


@dataclass
class Bar:
    """An example dataclass."""

    #: some documentation for attribute b
    b: str

    def set_b(self, compute_b: Callable[[], str]) -> None:
        """Set b from return of a given function.

        Args:
            compute_b (Callable[[], str]): function without arguments to determine b.
        """
        self.b = compute_b()


if __name__ == "__main__":
    # foo = Foo(1.0)  # example that would fail (but mypy can tell you in advance)
    foo = Foo(1)

    bar = Bar(b="excellent to each other")
    print(bar.b)

    bar.set_b(hello_world)
    print(bar.b)
