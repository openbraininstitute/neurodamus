"""git_version.py

Provides utilities for handling Git-style version strings (e.g., '9.0a-1485-g0d990513b').

Includes:
- GitVersion: class representing a version with base, commit count, and hash,
  supporting comparison operators.
- check_program_version: function to verify a program meets a minimum Git-version requirement.

Intended for use with programs that use Git-describe-style versioning, such as NEURON.
"""

import logging
import re
from functools import total_ordering

from packaging import version


@total_ordering
class GitVersion:
    """Represents a Git-style version string: base[-commits-gHASH]. Comparison considers
    base version first, then commit count.
    """

    _regex = re.compile(r"^([\w\.]+)(?:-(\d+)-g([0-9a-f]+))?$")
    # Fallback for shallow-clone version strings like "HEAD (2ac5cc71+) 2026-06-19"
    _regex_shallow = re.compile(r"^HEAD\s+\(([0-9a-f]+)\+?\)")

    def __init__(self, vstr):
        """Parse a version string like '9.0a-1485-g0d990513b' or '9.0a'.

        For shallow-clone builds that produce non-standard strings (e.g.
        'HEAD (2ac5cc71+) 2026-06-19') or empty strings, the version is
        treated as newer than any parseable release by using a very high
        base version.
        """
        if not vstr or not vstr.strip():
            # Empty version: assume recent dev build (shallow clone artifact)
            logging.warning("Empty NEURON version string; assuming recent dev build.")
            self.base = version.parse("99.99.99")
            self.commits = 0
            self.commit = ""
            return

        m = self._regex.match(vstr)
        if m:
            base, commits, commit = m.groups()
            self.base = version.parse(base)
            self.commits = int(commits or 0)
            self.commit = commit or ""
            return

        # Shallow-clone fallback: assume it's a recent dev build
        m_shallow = self._regex_shallow.match(vstr)
        if m_shallow:
            logging.warning(
                "Non-standard NEURON version string: '%s'; assuming recent dev build.", vstr
            )
            self.base = version.parse("99.99.99")
            self.commits = 0
            self.commit = m_shallow.group(1)
            return

        raise ValueError(f"Invalid version string: {vstr}")

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
