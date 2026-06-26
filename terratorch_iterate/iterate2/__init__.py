# terratorch_iterate.iterate2 package
# Re-export main so that `from terratorch_iterate.iterate2 import main` keeps
# working after iterate2.py was turned into a package directory.
from terratorch_iterate.iterate2._iterate2 import main  # noqa: F401
