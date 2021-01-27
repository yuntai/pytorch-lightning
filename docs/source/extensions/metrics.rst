.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.metrics import Metric

.. _metrics:

#######
Metrics
#######

``pytorch_lightning.metrics`` is a Metrics API created for easy metric development and usage in
PyTorch and PyTorch Lightning. It is rigorously tested for all edge cases and includes a growing list of
common metric implementations.

The metrics API provides ``update()``, ``compute()``, ``reset()`` functions to the user. The metric base class inherits
``nn.Module`` which allows us to call ``metric(...)`` directly. The ``forward()`` method of the base ``Metric`` class
serves the dual purpose of calling ``update()`` on its input and simultaneously returning the value of the metric over the
provided input.

.. warning::
    From v1.2 onward ``compute()`` will no longer automatically call ``reset()``,
    and it is up to the user to reset metrics between epochs, except in the case where the
    metric is directly passed to ``LightningModule``s ``self.log``.

These metrics work with DDP in PyTorch and PyTorch Lightning by default. When ``.compute()`` is called in
distributed mode, the internal state of each metric is synced and reduced across each process, so that the
logic present in ``.compute()`` is applied to state information from all processes.

The example below shows how to use a metric in your ``LightningModule``:

.. code-block:: python

    def __init__(self):
        ...
        self.accuracy = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        ...
        # log step metric
        self.log('train_acc_step', self.accuracy(preds, y))
        ...

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())


``Metric`` objects can also be directly logged, in which case Lightning will log
the metric based on ``on_step`` and ``on_epoch`` flags present in ``self.log(...)``.
If ``on_epoch`` is True, the logger automatically logs the end of epoch metric value by calling
``.compute()``.

.. note::
    ``sync_dist``, ``sync_dist_op``, ``sync_dist_group``, ``reduce_fx`` and ``tbptt_reduce_fx``
    flags from ``self.log(...)`` don't affect the metric logging in any manner. The metric class
    contains its own distributed synchronization logic.

    This however is only true for metrics that inherit the base class ``Metric``,
    and thus the functional metric API provides no support for in-built distributed synchronization
    or reduction functions.


.. code-block:: python

    def __init__(self):
        ...
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        ...
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.valid_acc(logits, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

.. note::

    If using metrics in data parallel mode (dp), the metric update/logging should be done
    in the ``<mode>_step_end`` method (where ``<mode>`` is either ``training``, ``validation``
    or ``test``). This is due to metric states else being destroyed after each forward pass,
    leading to wrong accumulation. In practice do the following:

    .. code-block:: python

        def training_step(self, batch, batch_idx):
            data, target = batch
            preds = self(data)
            ...
            return {'loss' : loss, 'preds' : preds, 'target' : target}

        def training_step_end(self, outputs):
            #update and log
            self.metric(outputs['preds'], outputs['target'])
            self.log('metric', self.metric)

This metrics API is independent of PyTorch Lightning. Metrics can directly be used in PyTorch as shown in the example:

.. code-block:: python

    from pytorch_lightning import metrics

    train_accuracy = metrics.Accuracy()
    valid_accuracy = metrics.Accuracy(compute_on_step=False)

    for epoch in range(epochs):
        for x, y in train_data:
            y_hat = model(x)

            # training step accuracy
            batch_acc = train_accuracy(y_hat, y)

        for x, y in valid_data:
            y_hat = model(x)
            valid_accuracy(y_hat, y)

    # total accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()

    # total accuracy over all validation batches
    total_valid_accuracy = valid_accuracy.compute()

.. note::

    Metrics contain internal states that keep track of the data seen so far.
    Do not mix metric states across training, validation and testing.
    It is highly recommended to re-initialize the metric per mode as
    shown in the examples above. For easy initializing the same metric multiple
    times, the ``.clone()`` method can be used:

    .. testcode::

        from pytorch_lightning.metrics import Accuracy

        def __init__(self):
            ...
            metric = Accuracy()
            self.train_acc = metric.clone()
            self.val_acc = metric.clone()
            self.test_acc = metric.clone()

.. note::

    Metric states are **not** added to the models ``state_dict`` by default.
    To change this, after initializing the metric, the method ``.persistent(mode)`` can
    be used to enable (``mode=True``) or disable (``mode=False``) this behaviour.

*******************
Metrics and devices
*******************

Metrics are simple subclasses of :class:`~torch.nn.Module` and their metric states behave
similar to buffers and parameters of modules. This means that metrics states should
be moved to the same device as the input of the metric:

.. code-block:: python

    from pytorch_lightning.metrics import Accuracy

    target = torch.tensor([1, 1, 0, 0], device=torch.device("cuda", 0))
    preds = torch.tensor([0, 1, 0, 0], device=torch.device("cuda", 0))

    # Metric states are always initialized on cpu, and needs to be moved to
    # the correct device
    confmat = Accuracy(num_classes=2).to(torch.device("cuda", 0))
    out = confmat(preds, target)
    print(out.device) # cuda:0

However, when **properly defined** inside a :class:`~pytorch_lightning.core.lightning.LightningModule`
, Lightning will automatically move the metrics to the same device as the data. Being
**properly defined** means that the metric is correctly identified as a child module of the
model (check ``.children()`` attribute of the model). Therefore, metrics cannot be placed
in native python ``list`` and ``dict``, as they will not be correctly identified
as child modules. Instead of ``list`` use :class:`~torch.nn.ModuleList` and instead of
``dict`` use :class:`~torch.nn.ModuleDict`.

.. testcode::

    from pytorch_lightning.metrics import Accuracy

    class MyModule(LightningModule):
        def __init__(self):
            ...
            # valid ways metrics will be identified as child modules
            self.metric1 = Accuracy()
            self.metric2 = nn.ModuleList(Accuracy())
            self.metric3 = nn.ModuleDict({'accuracy': Accuracy()})

        def training_step(self, batch, batch_idx):
            # all metrics will be on the same device as the input batch
            data, target = batch
            preds = self(data)
            ...
            val1 = self.metric1(preds, target)
            val2 = self.metric2[0](preds, target)
            val3 = self.metric3['accuracy'](preds, target)


*********************
Implementing a Metric
*********************

To implement your custom metric, subclass the base ``Metric`` class and implement the following methods:

- ``__init__()``: Each state variable should be called using ``self.add_state(...)``.
- ``update()``: Any code needed to update the state given any inputs to the metric.
- ``compute()``: Computes a final value from the state of the metric.

All you need to do is call ``add_state`` correctly to implement a custom metric with DDP.
``reset()`` is called on metric state variables added using ``add_state()``.

To see how metric states are synchronized across distributed processes, refer to ``add_state()`` docs
from the base ``Metric`` class.

Example implementation:

.. testcode::

    from pytorch_lightning.metrics import Metric

    class MyAccuracy(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape

            self.correct += torch.sum(preds == target)
            self.total += target.numel()

        def compute(self):
            return self.correct.float() / self.total

Metrics support backpropagation, if all computations involved in the metric calculation
are differentiable. However, note that the cached state is detached from the computational
graph and cannot be backpropagated. Not doing this would mean storing the computational
graph for each update call, which can lead to out-of-memory errors.
In practise this means that:

.. code-block:: python

    metric = MyMetric()
    val = metric(pred, target) # this value can be backpropagated
    val = metric.compute() # this value cannot be backpropagated

******************
Metric Arithmetics
******************

Metrics support most of python built-in operators for arithmetic, logic and bitwise operations.

For example for a metric that should return the sum of two different metrics, implementing a new metric is an overhead that is not necessary. 
It can now be done with:

.. code-block:: python

    first_metric = MyFirstMetric()
    second_metric = MySecondMetric()

    new_metric = first_metric + second_metric

``new_metric.update(*args, **kwargs)`` now calls update of ``first_metric`` and ``second_metric``. It forwards all positional arguments but 
forwards only the keyword arguments that are available in respective metric's update declaration.

Similarly ``new_metric.compute()`` now calls compute of ``first_metric`` and ``second_metric`` and adds the results up.

This pattern is implemented for the following operators (with ``a`` being metrics and ``b`` being metrics, tensors, integer or floats):

* Addition (``a + b``)
* Bitwise AND (``a & b``)
* Equality (``a == b``)
* Floordivision (``a // b``)
* Greater Equal (``a >= b``)
* Greater (``a > b``)
* Less Equal (``a <= b``)
* Less (``a < b``)
* Matrix Multiplication (``a @ b``)
* Modulo (``a % b``)
* Multiplication (``a * b``)
* Inequality (``a != b``)
* Bitwise OR (``a | b``)
* Power (``a ** b``)
* Substraction (``a - b``)
* True Division (``a / b``)
* Bitwise XOR (``a ^ b``)
* Absolute Value (``abs(a)``)
* Inversion (``~a``)
* Negative Value (``neg(a)``)
* Positive Value (``pos(a)``)

****************
MetricCollection
****************

In many cases it is beneficial to evaluate the model output by multiple metrics.
In this case the `MetricCollection` class may come in handy. It accepts a sequence
of metrics and wraps theses into a single callable metric class, with the same
interface as any other metric.

Example:

.. testcode::

    from pytorch_lightning.metrics import MetricCollection, Accuracy, Precision, Recall
    target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
    preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
    metric_collection = MetricCollection([
        Accuracy(),
        Precision(num_classes=3, average='macro'),
        Recall(num_classes=3, average='macro')
    ])
    print(metric_collection(preds, target))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    {'Accuracy': tensor(0.1250),
     'Precision': tensor(0.0667),
     'Recall': tensor(0.1111)}

Similarly it can also reduce the amount of code required to log multiple metrics
inside your LightningModule

.. code-block:: python

    def __init__(self):
        ...
        metrics = pl.metrics.MetricCollection(...)
        self.train_metrics = metrics.clone()
        self.valid_metrics = metrics.clone()

    def training_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.train_metrics(logits, y)
        # use log_dict instead of log
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prefix='train')

    def validation_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.valid_metrics(logits, y)
        # use log_dict instead of log
        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True, prefix='val')

.. note::

    `MetricCollection` as default assumes that all the metrics in the collection
    have the same call signature. If this is not the case, input that should be
    given to different metrics can given as keyword arguments to the collection.

.. autoclass:: pytorch_lightning.metrics.MetricCollection
    :noindex:

**********
Metric API
**********

.. autoclass:: pytorch_lightning.metrics.Metric
    :noindex:

***************************
Class vs Functional Metrics
***************************

The functional metrics follow the simple paradigm input in, output out. This means, they don't provide any advanced mechanisms for syncing across DDP nodes or aggregation over batches. They simply compute the metric value based on the given inputs.

Also, the integration within other parts of PyTorch Lightning will never be as tight as with the class-based interface.
If you look for just computing the values, the functional metrics are the way to go. However, if you are looking for the best integration and user experience, please consider also using the class interface.

**********************
Classification Metrics
**********************

Input types
-----------

For the purposes of classification metrics, inputs (predictions and targets) are split
into these categories (``N`` stands for the batch size and ``C`` for number of classes):

.. csv-table:: \*dtype ``binary`` means integers that are either 0 or 1
    :header: "Type", "preds shape", "preds dtype", "target shape", "target dtype"
    :widths: 20, 10, 10, 10, 10

    "Binary", "(N,)", "``float``", "(N,)", "``binary``\*"
    "Multi-class", "(N,)", "``int``", "(N,)", "``int``"
    "Multi-class with probabilities", "(N, C)", "``float``", "(N,)", "``int``"
    "Multi-label", "(N, ...)", "``float``", "(N, ...)", "``binary``\*"
    "Multi-dimensional multi-class", "(N, ...)", "``int``", "(N, ...)", "``int``"
    "Multi-dimensional multi-class with probabilities", "(N, C, ...)", "``float``", "(N, ...)", "``int``"

.. note::
    All dimensions of size 1 (except ``N``) are "squeezed out" at the beginning, so
    that, for example, a tensor of shape ``(N, 1)`` is treated as ``(N, )``.

When predictions or targets are integers, it is assumed that class labels start at 0, i.e.
the possible class labels are 0, 1, 2, 3, etc. Below are some examples of different input types

.. testcode::

    # Binary inputs
    binary_preds  = torch.tensor([0.6, 0.1, 0.9])
    binary_target = torch.tensor([1, 0, 2])

    # Multi-class inputs
    mc_preds  = torch.tensor([0, 2, 1])
    mc_target = torch.tensor([0, 1, 2])

    # Multi-class inputs with probabilities
    mc_preds_probs  = torch.tensor([[0.8, 0.2, 0], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    mc_target_probs = torch.tensor([0, 1, 2])

    # Multi-label inputs
    ml_preds  = torch.tensor([[0.2, 0.8, 0.9], [0.5, 0.6, 0.1], [0.3, 0.1, 0.1]])
    ml_target = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 0]])


Using the is_multiclass parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, you might have inputs which appear to be (multi-dimensional) multi-class
but are actually binary/multi-label - for example, if both predictions and targets are
integer (binary) tensors. Or it could be the other way around, you want to treat 
binary/multi-label inputs as 2-class (multi-dimensional) multi-class inputs.

For these cases, the metrics where this distinction would make a difference, expose the
``is_multiclass`` argument. Let's see how this is used on the example of 
:class:`~pytorch_lightning.metrics.classification.StatScores` metric.

First, let's consider the case with label predictions with 2 classes, which we want to
treat as binary.

.. testcode::

   from pytorch_lightning.metrics.functional import stat_scores

   # These inputs are supposed to be binary, but appear as multi-class
   preds  = torch.tensor([0, 1, 0])
   target = torch.tensor([1, 1, 0])

As you can see below, by default the inputs are treated
as multi-class. We can set ``is_multiclass=False`` to treat the inputs as binary - 
which is the same as converting the predictions to float beforehand.

.. doctest::

    >>> stat_scores(preds, target, reduce='macro', num_classes=2)
    tensor([[1, 1, 1, 0, 1],
            [1, 0, 1, 1, 2]])
    >>> stat_scores(preds, target, reduce='macro', num_classes=1, is_multiclass=False)
    tensor([[1, 0, 1, 1, 2]])
    >>> stat_scores(preds.float(), target, reduce='macro', num_classes=1)
    tensor([[1, 0, 1, 1, 2]])

Next, consider the opposite example: inputs are binary (as predictions are probabilities),
but we would like to treat them as 2-class multi-class, to obtain the metric for both classes.

.. testcode::

   preds  = torch.tensor([0.2, 0.7, 0.3])
   target = torch.tensor([1, 1, 0])

In this case we can set ``is_multiclass=True``, to treat the inputs as multi-class.

.. doctest::

    >>> stat_scores(preds, target, reduce='macro', num_classes=1)
    tensor([[1, 0, 1, 1, 2]])
    >>> stat_scores(preds, target, reduce='macro', num_classes=2, is_multiclass=True)
    tensor([[1, 1, 1, 0, 1],
            [1, 0, 1, 1, 2]])


Class Metrics (Classification)
------------------------------

Accuracy
~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.Accuracy
    :noindex:

AveragePrecision
~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.AveragePrecision
    :noindex:

ConfusionMatrix
~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.ConfusionMatrix
    :noindex:

F1
~~

.. autoclass:: pytorch_lightning.metrics.classification.F1
    :noindex:

FBeta
~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.FBeta
    :noindex:
    
IoU
~~~

.. autoclass:: pytorch_lightning.metrics.classification.IoU
    :noindex:

Hamming Distance
~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.HammingDistance
    :noindex:

Precision
~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.Precision
    :noindex:

PrecisionRecallCurve
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.PrecisionRecallCurve
    :noindex:

Recall
~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.Recall
    :noindex:

ROC
~~~

.. autoclass:: pytorch_lightning.metrics.classification.ROC
    :noindex:


StatScores
~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.classification.StatScores
    :noindex:


Functional Metrics (Classification)
-----------------------------------

accuracy [func]
~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.accuracy
    :noindex:

auc [func]
~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.classification.auc
    :noindex:


auroc [func]
~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.classification.auroc
    :noindex:


multiclass_auroc [func]
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.classification.multiclass_auroc
    :noindex:


average_precision [func]
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.average_precision
    :noindex:


confusion_matrix [func]
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.confusion_matrix
    :noindex:


dice_score [func]
~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.classification.dice_score
    :noindex:


f1 [func]
~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.f1
    :noindex:


fbeta [func]
~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.fbeta
    :noindex:

hamming_distance [func]
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.hamming_distance
    :noindex:

iou [func]
~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.iou
    :noindex:


roc [func]
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.roc
    :noindex:


precision [func]
~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.precision
    :noindex:


precision_recall [func]
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.precision_recall
    :noindex:


precision_recall_curve [func]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.precision_recall_curve
    :noindex:


recall [func]
~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.recall
    :noindex:

select_topk [func]
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.utils.select_topk
    :noindex:


stat_scores [func]
~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.stat_scores
    :noindex:


stat_scores_multiple_classes [func]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.classification.stat_scores_multiple_classes
    :noindex:


to_categorical [func]
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.utils.to_categorical
    :noindex:


to_onehot [func]
~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.utils.to_onehot
    :noindex:

******************
Regression Metrics
******************

Class Metrics (Regression)
--------------------------

ExplainedVariance
~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.regression.ExplainedVariance
    :noindex:


MeanAbsoluteError
~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.regression.MeanAbsoluteError
    :noindex:


MeanSquaredError
~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.regression.MeanSquaredError
    :noindex:


MeanSquaredLogError
~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_lightning.metrics.regression.MeanSquaredLogError
    :noindex:


PSNR
~~~~

.. autoclass:: pytorch_lightning.metrics.regression.PSNR
    :noindex:


SSIM
~~~~

.. autoclass:: pytorch_lightning.metrics.regression.SSIM
    :noindex:


R2Score
~~~~~~~

.. autoclass:: pytorch_lightning.metrics.regression.R2Score
    :noindex:

Functional Metrics (Regression)
-------------------------------

explained_variance [func]
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.explained_variance
    :noindex:


image_gradients [func]
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.image_gradients
    :noindex:


mean_absolute_error [func]
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.mean_absolute_error
    :noindex:


mean_squared_error [func]
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.mean_squared_error
    :noindex:


mean_squared_log_error [func]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.mean_squared_log_error
    :noindex:


psnr [func]
~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.psnr
    :noindex:


ssim [func]
~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.ssim
    :noindex:


r2score [func]
~~~~~~~~~~~~~~

.. autofunction:: pytorch_lightning.metrics.functional.r2score
    :noindex:


***
NLP
***

bleu_score [func]
-----------------

.. autofunction:: pytorch_lightning.metrics.functional.nlp.bleu_score
    :noindex:

********
Pairwise
********

embedding_similarity [func]
---------------------------

.. autofunction:: pytorch_lightning.metrics.functional.self_supervised.embedding_similarity
    :noindex:
