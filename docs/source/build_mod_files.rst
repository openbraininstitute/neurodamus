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
