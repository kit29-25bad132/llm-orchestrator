"""Consumer-root conftest for examples/support-triage.

Pytest's prepend import mode inserts *this directory* (the first parent of a
conftest without ``__init__.py``) into ``sys.path``, which is what makes
``import triage`` work when the suite is launched from any other directory
with the repository's pytest configuration as rootdir. Without this file the
suite fails with ``ModuleNotFoundError: No module named 'triage'``.
"""
