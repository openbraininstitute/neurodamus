"""Neurodamus is a software for handling neuronal simulation using neuron.

Copyright (c) 2018 Blue Brain Project, EPFL.
All rights reserved
"""

import logging
import sys

from neuron import h

from neurodamus import commands
from neurodamus.utils.cli import extract_arguments


def main():
    args = []
    try:
        args = extract_arguments(sys.argv)
    except ValueError:
        logging.exception()
        return 1

    return commands.neurodamus(args)


if __name__ == "__main__":
    # Returns exit code and calls MPI.Finalize
    h.quit(main())
