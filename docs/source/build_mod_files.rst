Building MOD Files
==================

MOD (or NMOD) files are a way of extending the behavior of NEURON.
See `Using NMODL Files <https://www.neuronsimulator.org/en/latest/courses/using_nmodl_files.html>`_ for more details.

When running neurodamus, these files need to be pre-compiled and available via the ``NRNMECH_LIB_PATH`` (for `NEURON`) and ``CORENEURONLIB``  (for `coreneuron`) environment variables.

Generally, there are 3 classes of MOD files considered within neurodamus; `internal support`, `shared support` and `scientific`.
Both the `support` ones are bundled with the neurodamus install, so the user only needs to supply the `scientific` ones.

To make compiling the union of all these files easier, neurodamus supplies the `neurodamus-compile-mods` tool.
This simplifies the creation of the compiled libraries, by allowing the user to provide only the scientific MOD files, and `neurodamus-compile-mods` will provide the supporting ones.

For instance:

.. code-block:: bash

   neurodamus-compile-mods \
    --input-dir some-mod-dir \
    --with-internal-mods \
    --output-type shell \
    --output-dir output

Will compile the files within `some-mod-dir` along with the support MOD files since `--with-internal-mods` was specified.
The compiled files will be put in the `output` directory.

Finally, the `--output-type` specifies what the tool will return on its ``STDOUT``.
If `shell` is chosen, then environment variable setting is output:

EX::

    NRNMECH_LIB_PATH=/path/to/libnrnmech.so
    SPECIALS_PATH=/some/path/to/where/the/specials/are


Alternatively, `json` can be chosen, at which point a `json` object is printed.

If a SONATA `circuit_config.json` file exists, it can be examined, and if there are `mechanisms_dir`, they will also be included in the compilation.

.. code-block:: bash

   neurodamus-compile-mods \
    --circuit-config circuit_config.json \
    [...]

If ``neurodamus-nrnivmodl`` is on the PATH, `neurodamus-compile-mods` will use that instead of nrnivmodl.

It is assumed that ``neurodamus-nrnivmodl`` provides the same interface as ``nrnivmodl``, but adds any custom requirements for extra libraries (ie: `libsonatareport`.)

The tool can handle files with the same content, and ones with duplicate names.
The former are deduplicated, so there aren't symbol collisions, and the later ones are chosen by "latest wins", in that ones that are later on the command line overwrite earlier ones.

Caching
-------

Compilation is skipped when nothing relevant has changed.
A build is identified by the names and contents of its MOD files, the compilation options (``--simulator``, ``--incflags``, ``--loadflags``) and the toolchain, so that upgrading NEURON does not reuse an ABI-incompatible library.
This identity is recorded in ``modules.json`` inside the output directory, and is written only after a successful compilation.

With ``--output-dir``, one build is cached per directory: reusing the directory for a different set of MOD files recompiles, and the previous build is lost.
With ``--cache-dir``, the output path is instead *derived* from the build identity, so unrelated builds coexist in one cache and identical ones are shared:

.. code-block:: bash

   eval "$(neurodamus-compile-mods \
    --circuit-config circuit_config.json \
    --with-internal-mods \
    --cache-dir /path/to/mod-cache \
    --output-type shell)"

   neurodamus simulation_config.json

Nothing is copied out of the cache; the exported ``NRNMECH_LIB_PATH``, ``SPECIALS_PATH`` and ``CORENEURONLIB`` point into the cache slot directly.

``--print-cache-path`` resolves the path a build *would* use and exits without compiling, which lets a job script decide whether a build step is needed at all.

Concurrent builds of the same slot are serialized with a lock file, and only the first one compiles; the others wait and then reuse the result.
A build that is interrupted leaves no ``modules.json``, so it is simply retried on the next invocation. The lock itself needs no cleanup: the kernel releases it when the holder exits, however it exits.

A build that *hangs* rather than exiting does hold the lock, so waiting is bounded by ``--lock-timeout`` (default 1800 seconds).
On expiry the command fails and names the lock file, rather than blocking for the remainder of the job's walltime.
Pass ``--lock-timeout 0`` to wait indefinitely.

.. note::

   Locking relies on ``flock``, which some parallel filesystems do not implement; Lustre, for instance, needs to be mounted with ``-o flock`` or ``-o localflock``.
   Where it is unavailable, a warning is emitted and the build proceeds unlocked, which is safe as long as the same slot is not built by two processes at once.
   ``--prune`` is conservative in that case and keeps any slot it cannot prove idle.

.. note::

   Compile before launching under MPI, as in the example above.
   Every rank calling ``neurodamus-compile-mods`` is correct but pointlessly serialized behind the lock.

Cache slots are never removed automatically. Prune the ones that have gone unused, which is safe to run while builds are in progress:

.. code-block:: bash

   neurodamus-compile-mods --cache-dir /path/to/mod-cache --prune --max-age-days 30
