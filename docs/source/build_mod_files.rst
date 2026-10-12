Building MOD Files
==================

MOD (or NMODL) files extend NEURON. They must be compiled before running
neurodamus and exposed through ``NRNMECH_LIB_PATH`` (NEURON) or
``CORENEURONLIB`` (CoreNEURON). See `Using NMODL Files
<https://www.neuronsimulator.org/en/latest/courses/using_nmodl_files.html>`_
for background.

``neurodamus-compile-mods`` combines the MOD files supplied by the user with
neurodamus support files. For example:

.. code-block:: bash

   neurodamus-compile-mods \
       --input-dir some-mod-dir \
       --with-internal-mods \
       --output-dir output \
       --output-type shell

``shell`` prints environment assignments such as
``NRNMECH_LIB_PATH=...`` and ``SPECIALS_PATH=...``. Use ``json`` for a JSON
object instead. A SONATA circuit config can provide additional mechanism
directories:

.. code-block:: bash

   neurodamus-compile-mods \
       --circuit-config circuit_config.json \
       --with-internal-mods \
       --output-dir output

The tool uses ``neurodamus-nrnivmodl`` when it is on ``PATH``; otherwise it
uses ``nrnivmodl``. Duplicate contents are deduplicated. For duplicate names,
the later input directory wins.

Caching
-------

Compilation is skipped when the MOD files, compiler options, simulator, and
toolchain are unchanged. ``modules.json`` records this identity and is written
only after a successful build.

Use ``--output-dir`` for a fixed build directory. Use ``--cache-dir`` to share
content-addressed builds without overwriting other MOD sets:

.. code-block:: bash

   eval "$(neurodamus-compile-mods \
       --circuit-config circuit_config.json \
       --with-internal-mods \
       --cache-dir /path/to/mod-cache \
       --output-type shell)"

   neurodamus simulation_config.json

The generated environment variables point directly into the cache; artifacts
are not copied per simulation. ``--print-cache-path`` shows the slot without
compiling.

Concurrent builds of the same slot are serialized. Waiting is limited to 1800
seconds by default; change it with ``--lock-timeout`` or use
``--lock-timeout 0`` to wait indefinitely. If the filesystem does not support
``flock``, the tool warns and builds without locking; do not build the same slot
concurrently in that case.

Compile once before ``mpirun`` rather than running the compiler independently
from every MPI rank. Remove unused cache slots with:

.. code-block:: bash

   neurodamus-compile-mods \
       --cache-dir /path/to/mod-cache \
       --prune \
       --max-age-days 30
