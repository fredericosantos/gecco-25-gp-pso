from operator import attrgetter
import random

def selTournamentElitism(
    individuals, k, tournsize, elitism_size=1, fit_attr="fitness"
) -> list:
    """Select *k* individuals from the *individuals* list using a combination
    of elitism and tournament selection.

    - First, the top `elitism_size` individuals (according to fitness)
    are chosen (elitism).
    - Then, the remaining `k - elitism_size` are chosen via standard
    tournament selection.

    :param individuals: A list of individuals to select from.
    :param k: The total number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param elitism_size: How many 'elite' individuals to always carry over.
    :param fit_attr: The attribute of individuals to use as selection criterion.
    :returns: A list of selected individuals (references to input individuals).
    """
    # 1) Elitism step: pick the top 'elitism_size' individuals
    elites = sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:elitism_size]

    # 2) Fill the rest via standard tournament
    remaining = []
    for _ in range(k - elitism_size):
        aspirants = random.choices(individuals, k=tournsize)
        chosen_aspirant = max(aspirants, key=attrgetter(fit_attr))
        remaining.append(chosen_aspirant)
    # Combine elites + tournament-chosen
    return elites, remaining