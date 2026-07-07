"""Database-backed job coordinator for CLAIMED.

A **job** is a whole batch/run; a **task** is one unit of work within it
(typically a file path).  Populate a job with tasks from a glob/directory of
files (``propagate_jobs``) and let many worker sessions pull the job's tasks
one-by-one (``work_jobs``), invoking any shell script as the worker.  The
database is the single source of truth and provides atomic claiming via row
locks, so workers can run across many nodes without a shared filesystem.

See :mod:`claimed.jobcoordinator.db` for the storage/claim logic and
:mod:`claimed.jobcoordinator.cli` for the command-line entry points wired into
``claimed propagate_jobs`` / ``claimed work_jobs``.
"""
