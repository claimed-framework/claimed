"""Core module for salutation."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

from typing import Optional
import click


def salutation():
    """Return salutation string."""
    return "Gruezi Mitenand"


# this is referred to in the setup.py as entrypoint to create a
# cli command on installation.
@click.command()
@click.argument("name")
@click.argument("surname")
@click.option("--pronouns", default=None, help="Be an ally.")
def formal_introduction(name: str, surname: str, pronouns: Optional[str]):
    """Introduce yourself and salute with full name."""
    introduction = f"My name is {name} {surname}"
    if pronouns:
        introduction += f" ({pronouns})"
    print(salutation())
    print(introduction.strip())
