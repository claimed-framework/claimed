# claimed.iterate.iterate2 package
# Re-export main so that `from claimed.iterate.iterate2 import main` keeps
# working after iterate2.py was turned into a package directory.
from claimed.iterate.iterate2._iterate2 import main  # noqa: F401
