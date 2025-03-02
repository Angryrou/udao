============
Introduction
============

Objective
---------
The `udao` Python library aims to solve a Multi-objective Optimization (MOO)
problem given the user-defined optimization problem and datasets.

The diagram of the UDAO pipeline is as follows.

.. image:: ../images/udao-io3.png
  :width: 800
  :alt: Diagram of the UDAO pipeline


Modules
-------

There are three main components in the UDAO pipeline:

* :doc:`Data processing <../user_guide/data_processing>`: The data preprocessing component aims to process the input datasets for training the models. The data processing step will also be used to process data on the fly during the optimization.
* :doc:`Modeling <../user_guide/model>`: The model training component trains and evaluates models that will act as functions for the optimization component. It is based on the `Lightning` library.
* :doc:`Optimization <../user_guide/optimization>`: The optimization component then aims to solve the MOO problem given the user-defined optimization problem.

The following diagram shows the relationship between the three components.

.. image:: ../images/full_pipeline.svg
  :width: 800
  :alt: the UDAO components and their interactions
