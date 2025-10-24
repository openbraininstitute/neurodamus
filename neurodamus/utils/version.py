"""git_version.py

Provides utilities for handling Git-style version strings (e.g., '9.0a-1485-g0d990513b').

Includes:
- GitVersion: class representing a version with base, commit count, and hash,
  supporting comparison operators.
- check_program_version: function to verify a program meets a minimum Git-version requirement.

Intended for use with programs that use Git-describe-style versioning, such as NEURON.
"""

import re
from functools import total_ordering

from packaging import version


def check_environment():
    """Ensure programs in the environment (like neuron) are compatible with Neurodamus."""
    import neuron

    # Neurodamus requires NEURON >= 9.0a-1485-g0d990513b because
    # the layout of `report.conf` changed in this commit.
    req_version = GitVersion("9.0a-1485-g0d990513b")

    curr_neuron_version = GitVersion(neuron.__version__)
    if curr_neuron_version < req_version:
        raise RuntimeError(f"NEURON >= {req_version} required, found {curr_neuron_version}")


@total_ordering
class GitVersion:
    """Represents a Git-style version string: base[-commits-gHASH]. Comparison considers
    base version first, then commit count.
    """

    _regex = re.compile(r"^([\w\.]+)(?:-(\d+)-g([0-9a-f]+))?$")

    def __init__(self, vstr):
        """Parse a version string like '9.0a-1485-g0d990513b' or '9.0a'."""
        m = self._regex.match(vstr)
        if not m:
            raise ValueError(f"Invalid version string: {vstr}")
        base, commits, commit = m.groups()
        self.base = version.parse(base)
        self.commits = int(commits or 0)
        self.commit = commit or ""

    def __eq__(self, other):
        return (self.base, self.commits) == (other.base, other.commits)

    def __lt__(self, other):
        if self.base != other.base:
            return self.base < other.base
        return self.commits < other.commits

    def __hash__(self):
        return hash((self.base, self.commits))

    def __str__(self):
        return f"{self.base}-{self.commits}-g{self.commit}" if self.commit else str(self.base)
