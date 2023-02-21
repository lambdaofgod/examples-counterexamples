import logging

from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    HyperParameterOptimizer,
    RandomSearch,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
    GridSearch,
    Objective
)


def job_complete_callback(
    job_id,  # type: str
    objective_value,  # type: float
    objective_iteration,  # type: int
    job_parameters,  # type: dict
    top_performance_job_id,  # type: str
):
    print(
        "Job completed!", job_id, objective_value, objective_iteration, job_parameters
    )
    if job_id == top_performance_job_id:
        print(
            "WOOT WOOT we broke the record! Objective reached {}".format(
                objective_value
            )
        )


# Connecting ClearML with the current process,
# from here on everything is logged automatically
# task = Task.init(
#    project_name="Hyper-Parameter Optimization",
#    task_name="Automatic Hyper-Parameter Optimization",
#    task_type=Task.TaskTypes.optimizer,
#    reuse_last_task_id=False,
# )

# experiment template to optimize in the hyper-parameter optimization
# args = {
#    "template_task_id": None,
#    "run_as_service": False,
# }
# args = task.connect(args)
# Set default queue name for the Training tasks themselves.
# later can be overridden in the UI
from clearml import PipelineController, Logger

def initialization_step(seed=10):
    import numpy as np
    import pandas as pd

    x = np.random.randn(1000)
    y = x ** 2
    return pd.DataFrame({"x": x, "y": y})


def loss_function(data, theta):
    import numpy as np

    return np.mean(np.abs(theta * data["x"] ** 2 - data["y"]))


def optimization_step(data, maxiter, popsize):
    from scipy.optimize import differential_evolution, Bounds
    import numpy as np

    def loss_function(data, theta):
        import numpy as np

        return np.mean(np.abs(theta * data["x"] ** 2 - data["y"]))

    def loss(theta):
        return loss_function(data, theta)

    optim_results = differential_evolution(
        loss, Bounds([-1], [1]), maxiter=maxiter, popsize=popsize
    )
    a = optim_results.x
    return a


def evaluation_step(data, a):
    from clearml import PipelineController, Logger
    def loss_function(data, theta):
        import numpy as np

        return np.mean(np.abs(theta * data["x"] ** 2 - data["y"]))

    loss_value = loss_function(data, a)
    print(f"loss={loss_value}")
    Logger.current_logger().report_scalar(title="loss", series="loss", value=loss_value, iteration=1)
    return loss_value


def setup_pipeline(maxiter, popsize):
    pipe = PipelineController(
    name="Hyperparam kata controller", project="Hyperparamater kata", version="0.0.1"
    )


    pipe.add_function_step(
        "initialization",
        initialization_step,
        function_kwargs=dict(seed=10),
        function_return=["data"],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        "optimization",
        optimization_step,
        function_kwargs=dict(data="${initialization.data}", maxiter=maxiter, popsize=popsize),
        function_return=["a"],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        "evaluation",
        evaluation_step,
        function_kwargs=dict(data="${initialization.data}", a="${optimization.a}"),
        function_return=["loss_value"],
        monitor_metrics=[("optimization", "loss_value")],
        cache_executed_step=True,
    )
    return pipe

import itertools

for (maxiter, popsize) in itertools.product([10, 100], [10, 100]):
    print(f"running pipe with (maxiter, popsize)={(maxiter, popsize)}")
    pipe = setup_pipeline(maxiter, popsize)
    pipe.set_default_execution_queue("default")
    pipe.start_locally(run_pipeline_steps_locally=True)

import ipdb; ipdb.set_trace()

#pipe.set_default_execution_queue("default")
#pipe.start_locally(run_pipeline_steps_locally=True)

