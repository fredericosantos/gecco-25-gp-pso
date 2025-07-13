import random
import warnings
from tqdm import TqdmExperimentalWarning
from deap.tools import Logbook
import mlflow
from tqdm.rich import trange

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def eaSimpleElite(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    elite_size=0,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    trange_=None,
):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    pop_size = len(population)
    if trange_ is None:
        trange_ = trange(
            ngen * pop_size, desc="Evolution Loop", position=0, leave=False
        )

    # Begin the generational process
    for gen in range(ngen):
        if gen != 0:
            # Select the next generation individuals
            elites, offspring = toolbox.select(population, pop_size)

            # Vary the pool of individuals
            offspring = varAnd(elites + offspring, toolbox, cxpb, mutpb)
        else:
            offspring = population

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit
            # update the progress bar by one step every 10 individuals
            if i % 10 == 0:
                trange_.update(10)

        # Replace the current population by the offspring
        if gen != 0:
            population[:] = elites + offspring[: pop_size - elite_size]

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)
            best_ind = halloffame[0]
        else:
            best_ind = population[0]

        if mlflow.active_run():
            log_mlflow_metrics(record, gen)
            metrics = {
                "best_ind_fitness": best_ind.fitness.values[0],
                "best_ind_size": len(best_ind),
                "best_ind_depth": best_ind.height,
            }
            mlflow.log_metrics(metrics, step=gen)
            mlflow.log_dict(
                {"best_velocity_function": str(best_ind)}, "best_velocity_function.yaml"
            )
    return population, logbook


def varAnd(population, toolbox, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def log_mlflow_metrics(record, step):
    """
    Logs the statistics from 'record' into MLflow for the given step (ngen).

    Args:
        record (dict): The dictionary containing fitness, size, and depth statistics.
        step (int): The generation number (ngen) to log metrics for.
    """
    metrics = {}

    # Flatten the dictionary for MLflow logging
    for category, stats in record.items():
        for stat_name, value in stats.items():
            metric_name = f"{category}_{stat_name}"
            metrics[metric_name] = float(value)  # Ensure it's a scalar

    # Log all metrics at once for the given step
    mlflow.log_metrics(metrics, step=step)
