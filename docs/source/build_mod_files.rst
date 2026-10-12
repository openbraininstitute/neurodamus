Building MOD Files
==================

MOD files extend NEURON and must be compiled before a neurodamus run. The
result is exposed through ``NRNMECH_LIB_PATH`` (NEURON) or ``CORENEURONLIB``
(CoreNEURON). See `Using NMODL Files
<https://www.neuronsimulator.org/en/latest/courses/using_nmodl_files.html>`_.

``neurodamus-compile-mods`` combines user MOD files with neurodamus support
files:

.. code-block:: bash

   neurodamus-compile-mods \
       --input-dir some-mod-dir \
       --with-internal-mods \
       --output-dir output \
       --output-type shell

``shell`` prints environment assignments; use ``json`` for a JSON object.
Use ``--circuit-config circuit_config.json`` to discover SONATA mechanism
directories. The tool prefers ``neurodamus-nrnivmodl`` over ``nrnivmodl``.
Duplicate contents are deduplicated; later input directories win duplicate
names.

Caching
-------

Compilation is skipped when MOD files, options, simulator, and toolchain match.
``--output-dir`` is always the materialized output location. Add ``--cache-dir``
to reuse compiled entries across output directories:

.. code-block:: bash

   eval "$(neurodamus-compile-mods \
       --circuit-config circuit_config.json --with-internal-mods \
       --output-dir /path/to/output --cache-dir /path/to/mod-cache \
       --output-type shell)"
   neurodamus simulation_config.json

A cache hit copies the entry into ``--output-dir`` and returns paths from that
directory. ``--clear-cache --max-age-days 30`` removes unused entries.

Concurrent builds of one slot are serialized. The default lock timeout is 300
seconds; use ``--lock-timeout 0`` to wait indefinitely. Unsupported ``flock``
filesystems warn and build unlocked, so the same slot must not be built
concurrently there. Compile once before ``mpirun`` rather than from every MPI
rank.
