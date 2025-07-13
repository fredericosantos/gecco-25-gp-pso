# import torch
import torch
from torch import Tensor

from deap import creator, base, tools, gp
from functools import partial
from src.benchmarks import BenchmarkFunction
from src.pso import SwarmBatch
from src.gp_primitives import Primitives
from src.gp_selections import selTournamentElitism


def calculate_distance(
    optimization_function: BenchmarkFunction, global_best_positions: Tensor
):
    optimum = optimization_function.optimum
    distance = torch.linalg.norm(optimum - global_best_positions, dim=1)
    return distance


def calc_fitness(
    vel_func: callable,
    optimization_funcs: list[BenchmarkFunction],
    pso_n_particles: int,
    pso_n_iterations: int,
    device: str,
    vel_func_str: str | None = None,
) -> float:
    """
    Calculate the fitness of the function based on the median distance from the global best
    positions of the optimization functions.
    If `func` is None, uses standard pso velocity function.
    """
    fitness = []
    ind_nan_count = 0
    for optim_func in optimization_funcs:
        swarm_batch, gbest_fitness = get_globalbest(
            vel_func,
            pso_n_particles,
            pso_n_iterations,
            device,
            vel_func_str,
            optim_func,
        )
        v = swarm_batch.velocities
        if torch.isnan(v).any():
            # print("Nan values found in fitness")
            ind_nan_count += 1
            return float("inf")

        optim_func_fitness = torch.median(gbest_fitness) + torch.std(gbest_fitness)
        fitness.append(optim_func_fitness)

    fitness = torch.stack(fitness)
    return fitness.mean().item()


def get_globalbest(
    vel_func, pso_n_particles, pso_n_iterations, device, vel_func_str, optim_func
):
    swarm_batch = SwarmBatch(
        benchmark_func=optim_func,
        velocity_function=vel_func,
        num_particles=pso_n_particles,
        device=device,
        vel_func_str=vel_func_str,
    )
    swarm_batch.optimize(pso_n_iterations)
    gbest_pos = swarm_batch.gbest_positions
    gbest_fitness = calculate_distance(optim_func, gbest_pos)
    return swarm_batch, gbest_fitness


class ExperimentGP:
    def __init__(
        self,
        optimization_funcs: list[BenchmarkFunction],
        pop_size: int,
        elitism: float,
        tourn_size: int,
        hall_of_fame_size: int,
        min_init_depth: int,
        max_init_depth: int,
        max_depth_limit: int,
        min_mut_depth: int,
        max_mut_depth: int,
        pso_n_particles: int,
        pso_n_iterations: int,
        device: str,
    ):
        self.optimization_funcs = optimization_funcs
        self.pop_size = pop_size
        assert 0 <= elitism <= 1, f"Elitism ({elitism}) must be between 0 and 1"
        self.elitism = elitism
        self.elitism_size = max(1 if elitism > 0 else 0, int(pop_size * elitism))
        assert 0 <= hall_of_fame_size <= pop_size, (
            f"Hall of fame size ({hall_of_fame_size}) must be between 0 and pop_size ({pop_size})"
        )
        self.hall_of_fame_size = hall_of_fame_size
        assert 0 <= tourn_size <= pop_size, (
            f"Tournament size ({tourn_size})must be between 0 and pop_size"
        )
        self.tourn_size = tourn_size
        self.min_init_depth = min_init_depth
        self.max_init_depth = max_init_depth
        self.max_depth_limit = max_depth_limit
        self.min_mut_depth = min_mut_depth
        self.max_mut_depth = max_mut_depth
        self.pso_n_particles = pso_n_particles
        self.pso_n_iterations = pso_n_iterations
        self.pso_bounds = optimization_funcs[0].bounds
        self.device = device

        # TODO add pi and dimensions
        self._pset_args_list = [
            "positions",
            "velocity",
            "gbest",
            "pbest",
            "center",
            "magnitude",
            "dispersion",
            # "pi",
            "num_particles",
        ]
        self.pset_args = {f"ARG{i}": arg for i, arg in enumerate(self._pset_args_list)}
        self.prim = Primitives(self.device)
        self.pset = self.get_primitive_set()
        self.toolbox = self.get_toolbox()
        self.population = self.toolbox.population(n=pop_size)
        self.hof = tools.HallOfFame(self.hall_of_fame_size)

    def calculate_fitness(self, *args, **kwargs) -> float:
        return calc_fitness(*args, **kwargs)

    def get_toolbox(self) -> base.Toolbox:
        toolbox = base.Toolbox()
        toolbox.register(
            "gp_expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.min_init_depth,
            max_=self.max_init_depth,
        )
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.gp_expr
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        def eval_individual(individual) -> tuple[float]:
            vel_func = toolbox.compile(expr=individual)
            fitness = calc_fitness(
                vel_func=vel_func,
                optimization_funcs=self.optimization_funcs,
                pso_n_particles=self.pso_n_particles,
                pso_n_iterations=self.pso_n_iterations,
                vel_func_str=str(individual),
                device=self.device,
            )
            return (fitness,)

        toolbox.register("evaluate", eval_individual)
        toolbox.register(
            "select",
            selTournamentElitism,
            tournsize=self.tourn_size,
            elitism_size=self.elitism_size,
        )
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register(
            "expr_mut",
            gp.genGrow,
            min_=self.min_mut_depth,
            max_=self.max_mut_depth,
        )
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate(
            "mate",
            gp.staticLimit(key=lambda ind: ind.height, max_value=self.max_depth_limit),
        )
        toolbox.decorate(
            "mutate",
            gp.staticLimit(key=lambda ind: ind.height, max_value=self.max_depth_limit),
        )

        return toolbox

    # TODO: add a way to ^dimensions or ^1/dimensions
    def get_primitive_set(self) -> gp.PrimitiveSet:
        pset = gp.PrimitiveSet("main", len(self.pset_args))
        pset.renameArguments(**self.pset_args)
        pset.addPrimitive(self.prim.add_array, 2, "add")  # noqa
        pset.addPrimitive(self.prim.sub_array, 2, "sub")  # noqa
        pset.addPrimitive(self.prim.mul_array, 2, "mul")  # noqa
        pset.addPrimitive(self.prim.div_array, 2, "div")  # noqa
        # pset.addPrimitive(self.prim.neg_array, 1, "neg")  # noqa
        # pset.addPrimitive(self.prim.exp_array, 1, name="exp")  # noqa
        # pset.addPrimitive(self.prim.square_array, 1, name="square")  # noqa
        # pset.addPrimitive(self.prim.sqrt_array, 1, name="sqrt")  # noqa
        # pset.addPrimitive(self.prim.abs_array, 1, name="abs")  # noqa
        pset.addPrimitive(self.prim.inv_array, 1, name="inv")  # noqa
        pset.addPrimitive(self.prim.cos_array, 1, name="cos")  # noqa
        # pset.addPrimitive(self.prim.sin_array, 1, name="sin")  # noqa
        # pset.addPrimitive(self.prim.relu_array, 1, name="relu")  # noqa
        # pset.addPrimitive(self.prim.kill_array, 1, name="kill")  # noqa
        # pset.addPrimitive(self.prim.norm_array, 2, name="norm")  # noqa

        # c = partial(
        #     self.prim.torch_rand_uniform, self.pso_bounds[0], self.pso_bounds[1]
        # )

        # FIXME changed pso_bounds to -1, 1
        c = partial(self.prim.torch_rand_uniform, -1, 1)

        pset.addEphemeralConstant("r1", c)
        pset.addEphemeralConstant("r2", c)
        pset.addEphemeralConstant("r3", c)
        return pset
