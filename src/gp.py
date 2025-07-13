import random
from deap import creator, base, tools, gp
import torch
import typer
import numpy as np
import mlflow
from src.benchmarks import BenchmarkFunction, PARTITIONS, generate_benchmark_instances
from src.gp_class import ExperimentGP, calculate_distance, calc_fitness
from datetime import datetime
from tqdm.rich import tqdm, trange
from src.ea_simple import eaSimpleElite
from src.paper_functions import psog3, psocd1, psodisp2
from src.custom_velocity_formula import make_custom_velocity_from_yaml

# Create only once
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

app = typer.Typer(pretty_exceptions_enable=True)


@app.command()
def run(
    runs: int = 30,
    pop_size: int = typer.Option(50, help="Population size for GP."),
    elitism: float = typer.Option(0.1, help="Elitism rate."),
    hall_of_fame_size: int = typer.Option(1, help="Hall of fame size."),
    tourn_size: int = typer.Option(2, help="Tournament size for selection."),
    n_generations: int = typer.Option(
        100, help="Number of generations for GP evolution."
    ),
    n_particles: int = typer.Option(100, help="Number of particles in the PSO swarm."),
    n_iterations: int = typer.Option(
        100, help="Number of PSO iterations for each GP individual's evaluation."
    ),
    n_problems: int = typer.Option(100, help="Number of problems per benchmark."),
    dim: int = typer.Option(2, help="Dimension of the problem space."),
    lower_bound: float = typer.Option(-5.0, help="Lower bound for search space."),
    upper_bound: float = typer.Option(5.0, help="Upper bound for search space."),
    lower_optimum_bound: float = typer.Option(-2.0, help="Lower bound for optimum."),
    upper_optimum_bound: float = typer.Option(2.0, help="Upper bound for optimum."),
    enable_optimum_bounds_testing: bool = typer.Option(
        True, help="Use bounds testing."
    ),
    min_init_depth: int = typer.Option(1, help="Minimum depth of initial trees."),
    max_init_depth: int = typer.Option(3, help="Maximum depth of initial trees."),
    max_depth_limit: int = typer.Option(10, help="Maximum depth of trees."),
    min_mut_depth: int = typer.Option(1, help="Minimum depth of mutation."),
    max_mut_depth: int = typer.Option(3, help="Maximum depth of mutation."),
    cxpb: float = typer.Option(0.7, help="Crossover probability."),
    mutpb: float = typer.Option(0.3, help="Mutation probability."),
    seed: int = typer.Option(None, help="Random seed."),
    partition: list[int] = typer.Option(
        [1, 2, 3, 4, 5], help="Benchmark partition, [1, 2, 3, 4, 5]"
    ),
):
    trange_ = trange(
        n_generations * pop_size * len(partition) * runs,
        desc="Total Evaluations",
        position=0,
        leave=False,
    )
    if True:
        assert set(partition).issubset({1, 2, 3, 4, 5}), (
            "Benchmark combination must be a subset of [1, 2, 3, 4, 5]"
        )
        assert len(partition) <= 5, (
            "Benchmark combination must be a list of integers with length <= 5"
        )
        assert len(partition) == len(set(partition)), (
            "Benchmark combination must be a list of unique integers"
        )
        assert pop_size > 0, "Population size must be a positive integer"
        assert elitism >= 0 and elitism <= 1, "Elitism must be a float between 0 and 1"
        assert hall_of_fame_size > 0, "Hall of fame size must be a positive integer"
        assert tourn_size > 0, "Tournament size must be a positive integer"
        assert n_generations > 0, "Number of generations must be a positive integer"
        assert n_particles > 0, "Number of particles must be a positive integer"
        assert n_iterations > 0, "Number of iterations must be a positive integer"
        assert n_problems > 0, "Number of problems must be a positive integer"
        assert dim > 0, "Dimension must be a positive integer"
        assert lower_bound < upper_bound, "Lower bound must be less than upper bound"
        assert lower_optimum_bound < upper_optimum_bound, (
            "Lower optimum bound must be less than upper optimum bound"
        )
        assert min_init_depth > 0, "Minimum initial depth must be a positive integer"
        assert max_init_depth > min_init_depth, (
            "Maximum initial depth must be greater than minimum initial depth"
        )
        assert max_depth_limit > 0, "Maximum depth limit must be a positive integer"
        assert min_mut_depth > 0, "Minimum mutation depth must be a positive integer"
        assert max_mut_depth > min_mut_depth, (
            "Maximum mutation depth must be greater than minimum mutation depth"
        )
        assert cxpb >= 0 and cxpb <= 1, (
            "Crossover probability must be a float between 0 and 1"
        )
        assert mutpb >= 0 and mutpb <= 1, (
            "Mutation probability must be a float between 0 and 1"
        )
        assert seed is None or isinstance(seed, int), "Seed must be an integer or None"
    for c in partition:
        for i in range(runs):
            experiment(
                pop_size=pop_size,
                elitism=elitism,
                hall_of_fame_size=hall_of_fame_size,
                tourn_size=tourn_size,
                n_generations=n_generations,
                n_particles=n_particles,
                n_iterations=n_iterations,
                n_problems=n_problems,
                dim=dim,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lower_optimum_bound=lower_optimum_bound,
                upper_optimum_bound=upper_optimum_bound,
                enable_optimum_bounds_testing=enable_optimum_bounds_testing,
                min_init_depth=min_init_depth,
                max_init_depth=max_init_depth,
                max_depth_limit=max_depth_limit,
                min_mut_depth=min_mut_depth,
                max_mut_depth=max_mut_depth,
                cxpb=cxpb,
                mutpb=mutpb,
                seed=seed,
                benchmark_comb=c,
                verbose=False,
                trange_=trange_,
            )


def experiment(
    pop_size: int = 50,
    elitism: float = 0.1,
    hall_of_fame_size: int = 1,
    tourn_size: int = 2,
    n_generations: int = 100,
    n_particles: int = 100,
    n_iterations: int = 100,
    n_problems: int = 100,
    dim: int = 30,
    lower_bound: float = -5.0,
    upper_bound: float = 5.0,
    lower_optimum_bound: float = -2.0,
    upper_optimum_bound: float = 2.0,
    enable_optimum_bounds_testing: bool = True,
    min_init_depth: int = 1,
    max_init_depth: int = 3,
    max_depth_limit: int = 10,
    min_mut_depth: int = 1,
    max_mut_depth: int = 3,
    cxpb: float = 0.7,
    mutpb: float = 0.3,
    seed: int | None = None,
    verbose: bool = True,
    benchmark_comb: int = 1,
    trange_=None,
):
    """
    Runs the GP+PSO experiment with the specified parameters.
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Benchmarks
    train_benchmarks = PARTITIONS[benchmark_comb]["train"]
    test_benchmarks = PARTITIONS[benchmark_comb]["test"]

    # Combination

    test_optimum_bounds = (
        (lower_optimum_bound, upper_optimum_bound)
        if enable_optimum_bounds_testing
        else (lower_bound, upper_bound)
    )

    dataset_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    params_dict = {
        "pop_size": pop_size,
        "elitism": elitism,
        "tourn_size": tourn_size,
        "n_generations": n_generations,
        "n_particles": n_particles,
        "n_iterations": n_iterations,
        "n_problems": n_problems,
        "dim": dim,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_optimum_bound": lower_optimum_bound,
        "upper_optimum_bound": upper_optimum_bound,
        "lower_optimum_bound_test": test_optimum_bounds[0],
        "upper_optimum_bound_test": test_optimum_bounds[1],
        "max_depth_limit": max_depth_limit,
        "min_init_depth": min_init_depth,
        "max_init_depth": max_init_depth,
        "min_mut_depth": min_mut_depth,
        "max_mut_depth": max_mut_depth,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "seed": seed,
        "train_benchmarks": train_benchmarks,
        "test_benchmarks": test_benchmarks,
        "dataset_id": dataset_id,
    }

    # MLFlow start run
    mlflow.set_experiment("GP+PSO [train]")
    mlflow.start_run()
    mlflow.log_params(params_dict | {"run_type": "gp+pso train"})

    # Optimzation functions
    train_benchmarks = generate_benchmark_instances(
        train_benchmarks,
        n_problems,
        dim,
        lower_bound,
        upper_bound,
        lower_optimum_bound,
        upper_optimum_bound,
        seed,
        device,
    )

    gp_exp = ExperimentGP(
        optimization_funcs=train_benchmarks,
        pop_size=pop_size,
        elitism=elitism,
        tourn_size=tourn_size,
        hall_of_fame_size=hall_of_fame_size,
        min_init_depth=min_init_depth,
        max_init_depth=max_init_depth,
        max_depth_limit=max_depth_limit,
        min_mut_depth=min_mut_depth,
        max_mut_depth=max_mut_depth,
        pso_n_particles=n_particles,
        pso_n_iterations=n_iterations,
        device=device,
    )
    arguments = gp_exp.pset.arguments
    context = list(gp_exp.pset.context.keys())
    context.remove("__builtins__")
    mlflow.log_params({"arguments": arguments, "context": context})

    # Prepare stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(lambda ind: len(ind))
    stats_depth = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(
        fitness=stats_fit, size=stats_size, depth=stats_depth
    )
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    if verbose:
        typer.echo(f"Starting evolution on {device} with {seed = }", color="green")
        typer.echo(
            f"GP config:\n{max_depth_limit = } -- tree init depth (min, max) = {(min_init_depth, max_init_depth)} -- tree mutation depth (min, max) = {(min_mut_depth, max_mut_depth)}"
        )
        typer.echo(f"{pop_size = }, {n_generations = }, {cxpb = }, {mutpb = }")
        typer.echo(f"PSO config:\n{n_particles = }, {n_iterations = }")
        typer.echo(f"Train Benchmarks = {train_benchmarks}")
        typer.echo(
            f"Benchmark config: {n_problems = } -- space bounds = {(upper_bound, lower_bound)} -- global optimum bounds = {(lower_optimum_bound, upper_optimum_bound)} -- {dim = }"
        )

    # DEAP's simple evolutionary algorithm
    population, logbook = eaSimpleElite(
        gp_exp.population,
        gp_exp.toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=n_generations,
        elite_size=gp_exp.elitism_size,
        stats=mstats,
        halloffame=gp_exp.hof,
        verbose=False,
        trange_=trange_,
    )

    best_ind = gp_exp.hof[0]
    best_ind_func = gp_exp.toolbox.compile(expr=best_ind)
    train_fitness = best_ind.fitness.values[0]
    if verbose:
        typer.echo(
            f"Best individual w/ size {len(best_ind)} and max depth ({best_ind.height}):\n{best_ind}"
        )
        typer.echo(f"Best training fitness: {train_fitness}")

        typer.echo("\nFinal log stats:")
        typer.echo(logbook)

    fit = {
        "fitness": train_fitness,
        "train_fitness": train_fitness,
    }
    mlflow.log_metrics(metrics=fit)

    # Test the best individual
    test(
        n_problems=n_problems,
        dim=dim,
        bounds=(lower_bound, upper_bound),
        pso_n_particles=n_particles,
        pso_n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
        device=device,
        test_optimum_bounds=test_optimum_bounds,
        params_dict=params_dict,
        vel_func=best_ind_func,
        benchmark_comb=benchmark_comb,
        from_train=True,
    )


def test(
    n_problems: int,
    dim: int,
    bounds: tuple[float, float],
    test_optimum_bounds: tuple[float, float],
    vel_func: callable,
    pso_n_particles: int,
    pso_n_iterations: int,
    benchmark_comb: int,
    device: str,
    seed: int | None = None,
    params_dict: dict | None = None,
    exp_name: str | None = None,
    tags: dict = {},
    verbose: bool = False,
    from_train: bool = False,
    nested: bool = False,
):
    if seed is None:
        seed = np.random.randint(0, 1000)
    # if no mlflow run is active, start one
    if not from_train:
        if not nested:
            exp_name = "Custom Func vs Rest" if exp_name is None else exp_name
            mlflow.set_experiment(exp_name)
        mlflow.start_run(nested=nested)
        mlflow.log_params(tags)
    train_benchmarks = PARTITIONS[benchmark_comb]["train"]
    test_benchmarks = PARTITIONS[benchmark_comb]["test"]
    if params_dict is None:
        dataset_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        params_dict = {
            "pso_n_particles": pso_n_particles,
            "pso_n_iterations": pso_n_iterations,
            "n_problems": n_problems,
            "dim": dim,
            "lower_bound": bounds[0],
            "upper_bound": bounds[1],
            "lower_optimum_bound": test_optimum_bounds[0],
            "upper_optimum_bound": test_optimum_bounds[1],
            "seed": seed,
            "test_benchmarks": test_benchmarks,
            "train_benchmarks": train_benchmarks,
            "dataset_id": dataset_id,
        }
        mlflow.log_params(params_dict | {"run_type": "test"})
    test_bms = [
        BenchmarkFunction(
            benchmark_name=benchmark_name,
            dimensions=dim,
            bounds=bounds,
            num_problems=n_problems,
            device=device,
            optimum_bounds=test_optimum_bounds,
            seed=seed,
        )
        for benchmark_name in test_benchmarks
    ]
    test_bms = generate_benchmark_instances(
        test_benchmarks,
        n_problems,
        dim,
        bounds[0],
        bounds[1],
        test_optimum_bounds[0],
        test_optimum_bounds[1],
        seed,
        device,
    )
    shared_params = dict(
        optimization_funcs=test_bms,
        pso_n_particles=pso_n_particles,
        pso_n_iterations=pso_n_iterations,
        device=device,
    )
    test_fitness = calc_fitness(vel_func, **shared_params)
    test_fitness_pso = calc_fitness(None, **shared_params)
    test_fitness_g3 = calc_fitness(psog3, **shared_params)
    test_fitness_cd1 = calc_fitness(psocd1, **shared_params)
    test_fitness_disp2 = calc_fitness(psodisp2, **shared_params)

    if verbose:
        typer.echo(f"Test Benchmarks = {test_benchmarks}")

    test_fit = {
        "test_fitness": test_fitness,
        "test_fitness_pso": test_fitness_pso,
        "test_fitness_g3": test_fitness_g3,
        "test_fitness_cd1": test_fitness_cd1,
        "test_fitness_disp2": test_fitness_disp2,
    }
    mlflow.log_metrics(metrics=test_fit)
    mlflow.end_run()

    test_per_benchmark(
        vel_func=vel_func,
        params_dict=params_dict,
        tags=tags,
        test_bms=test_bms,
        pso_n_particles=pso_n_particles,
        pso_n_iterations=pso_n_iterations,
        device=device,
        test_fit=test_fit,
        exp_name=exp_name,
        from_train=from_train,
        nested=nested,
    )


def test_per_benchmark(
    vel_func: callable,
    params_dict: dict,
    test_bms: list[BenchmarkFunction],
    pso_n_particles: int,
    pso_n_iterations: int,
    device: str,
    test_fit: dict,
    tags: dict = {},
    exp_name: str | None = None,
    from_train: bool = False,
    nested: bool = False,
):
    fit_dict = {
        "gp+pso": test_fit["test_fitness"],
        "pso": test_fit["test_fitness_pso"],
        "psog3": test_fit["test_fitness_g3"],
        "psocd1": test_fit["test_fitness_cd1"],
        "psodisp2": test_fit["test_fitness_disp2"],
    }

    shared_params = dict(
        pso_n_particles=pso_n_particles,
        pso_n_iterations=pso_n_iterations,
        device=device,
    )

    for n, f in zip(
        fit_dict.keys(),
        [vel_func, None, psog3, psocd1, psodisp2],
    ):
        if not nested:
            exp_name_ = exp_name if exp_name is not None else f"{n.upper()} [test]"
            mlflow.set_experiment(exp_name_)
        mlflow.start_run(nested=nested)
        if not from_train:
            mlflow.log_params(tags)
        for test_func in test_bms:
            k = f"test_fit_{test_func.name}"
            fitness_func = calc_fitness(f, [test_func], **shared_params)
            mlflow.log_metrics({k: fitness_func})

        mlflow.log_params(params_dict | {"run_type": f"{n} test"})
        mlflow.log_metric("fitness", fit_dict[n])
        mlflow.end_run()


if __name__ == "__main__":
    app()
