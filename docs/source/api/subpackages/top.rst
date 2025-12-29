======================
Neurodamus Sub-Modules
======================

neurodamus.node
============================

.. automodule:: neurodamus.node
   :members:
   :undoc-members:
   :exclude-members: Neurodamus, Node

   .. rubric:: Classes

   .. autosummary::
      CircuitManager
      METypeEngine

neurodamus.target\_manager
==========================

.. automodule:: neurodamus.target_manager
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      TargetManager
      NodeSetReader
      NodesetTarget
      SerializedSections
      TargetPointList

neurodamus.stimulus\_manager
============================

.. automodule:: neurodamus.stimulus_manager
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      StimulusManager
      BaseStimulus
      OrnsteinUhlenbeck
      RelativeOrnsteinUhlenbeck
      ShotNoise
      RelativeShotNoise
      AbsoluteShotNoise
      Linear
      Hyperpolarizing
      RelativeLinear
      SubThreshold
      Noise
      Pulse
      Sinusoidal
      SEClamp

neurodamus.cell\_distributor
============================

.. automodule:: neurodamus.cell_distributor
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      CellDistributor
      CellManagerBase
      LoadBalance


neurodamus.connection
=====================

.. automodule:: neurodamus.connection
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      NetConType
      ReplayMode
      ConnectionBase
      Connection
      ArtificialStim
      SpontMinis
      InhExcSpontMinis
      ReplayStim


neurodamus.connection\_manager
==============================

.. automodule:: neurodamus.connection_manager
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      ConnectionSet
      ConnectionManagerBase
      SynapseRuleManager
      edge_node_pop_names

   .. autoclass:: SynapseRuleManager
      :members:
      :noindex:
      :inherited-members:


neurodamus.metype
=================

.. automodule:: neurodamus.metype
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      METype
      METypeManager


neurodamus.replay
=================

.. automodule:: neurodamus.replay
   :members:
   :undoc-members:

   .. rubric:: Classes

   .. autosummary::
      SpikeManager

neurodamus.gap\_junction
============================

.. automodule:: neurodamus.gap_junction
   :members:
   :undoc-members:

   .. rubric:: Classes

