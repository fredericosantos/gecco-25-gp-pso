import torch

"""
- PSOG3 = R_1 (global_best_i - x-i) - 0.75 R_2 * R_1 * x_i * (global_best_i)^2 - 0.25 * R_3 * R_2 * R_1 * x_i * global_best_i
- PSOCD1 = (global_best_i - x_i) - ((R^2)/d) - v_i
- PSODISP2 = (global_best_i - x_i) + (local_best_i - x_i) - d*x_i
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def psog3(
    positions,
    velocity,
    gbest,
    pbest,
    center,
    magnitude,
    dispersion,
    num_particles,
):
    """
    Computes PSOG3:
    g3 = R_1 * (gbest - positions) - 0.75 * R_2 * R_1 * positions * (gbest)^2 - 0.25 * R_3 * R_2 * R_1 * positions * gbest
    """
    R_1 = torch.rand_like(positions)
    R_2 = torch.rand_like(positions)
    R_3 = torch.rand_like(positions)

    return R_1 * (gbest - positions) - 0.75 * R_2 * R_1 * positions * (gbest ** 2) - 0.25 * R_3 * R_2 * R_1 * positions * gbest


def psocd1(
    positions,
    velocity,
    gbest,
    pbest,
    center,
    magnitude,
    dispersion,
    num_particles,
):
    """
    Computes PSOCD1:
    cd1 = (gbest - positions) - ((R_1^2) / center) - velocity
    """
    R_1 = torch.rand_like(positions)

    return (gbest - positions) - ((R_1 ** 2) / center) - velocity



def psodisp2(
    positions,
    velocity,
    gbest,
    pbest,
    center,
    magnitude,
    dispersion,
    num_particles,
):
    """
    Computes PSODISP2:
    disp2 = (gbest - positions) + (pbest - positions) - dispersion * positions
    """

    return (gbest - positions) + (pbest - positions) - dispersion * positions