from datetime import datetime
import numpy as np
import typer
from typing import List, Tuple, Optional
from rich.progress import track
from tqdm.rich import tqdm
from src.benchmarks import PARTITIONS, BenchmarkFunction
from src.gp_class import calc_fitness
from src.custom_velocity_formula import make_custom_velocity_from_yaml
import torch
from rich import print
import mlflow

from src.paper_functions import psocd1, psodisp2, psog3

app = typer.Typer(pretty_exceptions_enable=True)


def sorted_bounds(bounds: Tuple[float, float]) -> Tuple[float, float]:
    return tuple(sorted(bounds))


def create_function_name(run_id: str, dim: int, partition: int):
    function_name = f"X-{run_id[:4]}-{dim}"
    return function_name.upper()

def change_function_name(run_type: str | None, partition: int):
    if run_type is None:
        return None
    function_name = f"P{partition}-{run_type[2:]}D"
    return function_name.upper()

@app.command()
def run(
    experiment_name: str = typer.Option(
        "custom velocity", help="Name of the experiment to run"
    ),
    run_ids: list[str] = typer.Option(
        ..., help="Required run ID(s) for the velocity function"
    ),
    dims: list[int] = typer.Option(
        [30], help="List of dimensions to test", show_default=True
    ),
    n_runs: int = typer.Option(5, help="Number of repetitions for each setting"),
    n_problems: int = typer.Option(100, help="Number of test problems"),
    bounds: list[float, float] = typer.Option(
        [-5.0, 5.0], help="Bounds for the problem space (e.g., --bounds -5 5)"
    ),
    optimum_bounds: list[float, float] = typer.Option(
        [-2.0, 2.0], help="Bounds for the optimum space (e.g., --optimum-bounds -2 2)"
    ),
    n_particles: int = typer.Option(100, help="Number of particles"),
    n_iterations: int = typer.Option(100, help="Number of iterations"),
    device: Optional[str] = typer.Option(
        None, help="Torch device (e.g., 'cuda' or 'cpu')"
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    bounds = sorted_bounds(bounds)
    assert len(bounds) == 2, "Bounds must be a tuple of length 2"
    optimum_bounds = sorted_bounds(optimum_bounds)

    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

    runs_dimevo = mlflow.search_runs(
        experiment_names=["GP+PSO [train]"],
        filter_string=f"attributes.run_id IN ({','.join([f"'{run_id}'" for run_id in run_ids])})",
    )[["run_id", "params.dim", "params.test_benchmarks"]]

    inverse_dict = {str(v["test"]): k for k, v in PARTITIONS.items()}
    # create col in runs_dimevo with the partition name
    runs_dimevo["partition"] = runs_dimevo["params.test_benchmarks"].apply(
        lambda x: inverse_dict[x]
    )
    runs_dimevo["dim"] = runs_dimevo["params.dim"].astype(int)
    runs_dimevo["partition"] = runs_dimevo["partition"].astype(int)
    partition_x_runids = (
        runs_dimevo.groupby("partition")["run_id"].apply(list).to_dict()
    )
    runid_x_dim = runs_dimevo.set_index("run_id")["params.dim"].to_dict()
    for n in tqdm(range(n_runs), desc="Runs"):
        for d in dims:
            for p in PARTITIONS.keys():
                seed = np.random.randint(0, 10000)
                dataset_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                train_benchmarks = PARTITIONS[p]["train"]
                test_benchmarks = PARTITIONS[p]["test"]
                test_bms = [
                    BenchmarkFunction(
                        benchmark_name=benchmark_name,
                        dimensions=d,
                        bounds=bounds,
                        num_problems=n_problems,
                        device=device,
                        optimum_bounds=optimum_bounds,
                        seed=seed,
                    )
                    for benchmark_name in test_benchmarks
                ]
                shared_params = dict(
                    pso_n_particles=n_particles,
                    pso_n_iterations=n_iterations,
                    device=device,
                )
                params_dict = {
                    "pso_n_particles": n_particles,
                    "pso_n_iterations": n_iterations,
                    "n_problems": n_problems,
                    "dim": d,
                    "lower_bound": bounds[0],
                    "upper_bound": bounds[1],
                    "lower_optimum_bound": optimum_bounds[0],
                    "upper_optimum_bound": optimum_bounds[1],
                    "seed": seed,
                    "test_benchmarks": test_benchmarks,
                    "train_benchmarks": train_benchmarks,
                    "dataset_id": dataset_id,
                }

                func_name_to_callable = {
                    "pso": None,
                    "psog3": psog3,
                    "psocd1": psocd1,
                    "psodisp2": psodisp2,
                }

                fitness_algo = {
                    "pso": {},
                    "psog3": {},
                    "psocd1": {},
                    "psodisp2": {},
                }
                for name, call in func_name_to_callable.items():
                    mlflow.start_run(nested=True)
                    fitness_algo[name]["fitness"] = calc_fitness(
                        call, test_bms, **shared_params
                    )
                    for test_func in test_bms:
                        k = f"test_fit_{test_func.name}"
                        fitness_algo[name][k] = calc_fitness(
                            call, [test_func], **shared_params
                        )
                    mlflow.log_params(params_dict | {"run_type": f"{name}"})
                    mlflow.log_metrics(fitness_algo[name])
                    mlflow.end_run()

                if p not in partition_x_runids:
                    continue
                for id in partition_x_runids[p]:
                    custom_vel_func = make_custom_velocity_from_yaml(id, device)
                    function_name = create_function_name(id, runid_x_dim[id], p)
                    fitness_algo[function_name] = {}
                    mlflow.start_run(nested=True)
                    mlflow.log_params({"func_run_id": id})
                    mlflow.log_params(params_dict | {"run_type": function_name})
                    fitness_algo[function_name]["fitness"] = calc_fitness(
                        custom_vel_func, test_bms, **shared_params
                    )

                    for test_func in test_bms:
                        k = f"test_fit_{test_func.name}"
                        fitness_algo[function_name][k] = calc_fitness(
                            custom_vel_func, [test_func], **shared_params
                        )
                    mlflow.log_metrics(fitness_algo[function_name])
                    mlflow.end_run()
    mlflow.end_run()


if __name__ == "__main__":
    app()
