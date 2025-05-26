=======================
neurodamus.core package
=======================


.. automodule:: neurodamus.core
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      Neuron
      MPI
      NeuronWrapper
      ProgressBarRank0

   .. rubric:: Decorators

   .. autosummary::
      return_neuron_timings
      mpi_no_errors
      run_only_rank0


Sub-Modules
===========

.. autosummary::
   configuration
   random
   stimuli


Module API
==========

.. autoclass:: Neuron
   :members:

.. autoclass:: MPI

   *property* :py:attr:`rank` The id of the current MPI rank

   *property* :py:attr:`size` The number of MPI ranks

.. autoclass:: NeuronWrapper
   :members:

.. autoclass:: neurodamus.core._utils.ProgressBarRank0
   :members:

.. autoclass:: neurodamus.core._neuron._Neuron
   :members: h, load_dll, load_hoc, require, run_sim, section_in_stack


**Decorators**

.. autofunction:: neurodamus.core._utils.return_neuron_timings

.. autofunction:: neurodamus.core._utils.mpi_no_errors

.. autofunction:: neurodamus.core._utils.run_only_rank0
