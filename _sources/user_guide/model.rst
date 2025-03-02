===============================
Modeling the objective function
===============================

The objective function is what we want to minimize in the optimization module.
Here we aim at training a machine learning model from the training data prepared in :doc:`Data processing <../user_guide/data_processing>`, as the objective function to minimize.

The case of query plans
-----------------------

To model a latency function from a query plan graph and features, we provide two components:

* Embedders that embed the query plan graph and features into a vector space, based on the :py:class:`~udao.model.embedders.base_graph_embedder.BaseGraphEmbedder` class.
* Regressors that take tabular features concatened with the query plan embedding to output the predicted latency. The :py:class:`~udao.model.regressors.mlp.MLP` implements an MLP regressor.

Embedders
~~~~~~~~~
Several embedders are available in the embedders module. They all inherit from the :py:class:`~udao.model.embedders.base_graph_embedder.BaseGraphEmbedder` class.

Regressors
~~~~~~~~~~
The MLP regressor then takes the concatenation of the query plan embedding and the tabular features as input, and outputs the predicted latency.

The UdaoModel
~~~~~~~~~~~~~

The :py:class:`~udao.model.model.UdaoModel` class is a wrapper around the embedder and regressor.
It is used to train the model and predict the latency of a query plan.

Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~~
To train a UdaoModel, we use the :py:class:`~udao.model.module.UdaoModule` inheriting LightningModel from Pytorch Lightning to set up the training parameters.
We then use Pytorch Lightning's `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ to train the model.

Here is a minimal example of how to train a UdaoModel::

    # First process the data
    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": 1024,
            "op_groups": ["cbo", "op_enc", "type"],
            "type_embedding_dim": 8,
            "embedding_normalizer": None,
        },
        regressor_params={"n_layers": 2, "hidden_dim": 32, "dropout": 0.1},
    )
    module = UdaoModule(
        model,
        ["latency"],
        loss=WMAPELoss(),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])

    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=10,
        callbacks=[scheduler],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(batch_size),
        val_dataloaders=split_iterators["val"].get_dataloader(batch_size),
    )
