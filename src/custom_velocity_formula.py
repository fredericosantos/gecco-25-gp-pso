import mlflow
import torch
import yaml
from src.gp_primitives import Primitives
from torch import tensor, pi


VEL: str = """add(add(gbest, positions), sub(sub(pbest, positions), mul(dispersion, positions)))"""
VEL: str = """mul(sin(sin(sub(sin(sin(square(positions))), sin(sin(square(sin(add(velocity, inv(4.101810553492628))))))))), positions)"""

VEL: str = """sub(mul(sub(positions, gbest), tensor([-0.1687], device='cuda:0')),
  div(cos(add(mul(cos(add(sub(add(sub(mul(sub(add(sub(tensor([-0.1687], device='cuda:0'),
  tensor([4.9126], device='cuda:0')), tensor([-0.1687], device='cuda:0')), positions),
  tensor([-0.1687], device='cuda:0')), add(div(magnitude, tensor([-0.1687], device='cuda:0')),
  tensor([-0.1687], device='cuda:0'))), mul(div(num_particles, center), add(sub(tensor([-0.1687], device='cuda:0'), tensor([4.9126], device='cuda:0')), positions))), add(sub(add(sub(gbest,
  add(add(positions, gbest), tensor([-0.1687], device='cuda:0'))), mul(tensor([-0.1687], device='cuda:0'), cos(num_particles))), add(gbest, sub(tensor([-0.1687], device='cuda:0'),
  pi))), div(div(pbest, neg(pi)), neg(pi)))), div(num_particles, neg(pi)))), num_particles),
  mul(sub(add(sub(tensor([-0.1687], device='cuda:0'), tensor([4.9126], device='cuda:0')),
  tensor([-0.1687], device='cuda:0')), positions), cos(num_particles)))), pi))"""

# VEL_STR = """mul(tensor([-0.4104], device='cuda:0'), add(mul(gbest, div(sub(add(inv(sub(velocity,
#   mul(cos(tensor([-0.1484], device='cuda:0')), positions))), div(positions, sub(add(cos(tensor([-0.1261],
#   device='cuda:0')), div(positions, pbest)), magnitude))), positions), mul(magnitude,
#   positions))), positions))"""


# b623458c06ea46e7a7fa2116626595b0 - exploits constraints [-2, 2]
# 00fd1bb26b3c4ea89032f12e6818514a - exploits constraints [-2, 2]
# 8894451cec7a4b14b26d00e164ff9a93 - exploits constraints [-2, 2]
# ccd4c24301e8422a957372e5fa944d4e - explorts constraints [-2, 2]
# 037d1a7bd04440a491fb9fb0e1f67d81 - exploits constraints [-2, 2]
# b302832094ec42c1bcd775405af1876e [dim=100]
# quite interesting, shoots from center mass!
# 7604f4e678f34a4f9ca299eef4d4dc3e - exploits constraints [-2, 2]
# 1b1e2825947b42bdabe751c5160f1eb7 ✅
# 8a131b84402f4b56bd38308ab5bff51e ✅
# bef55e6aa98d4647a1230cfdf2c95657 ✅
# 00ccd6f9c1084f91a0e6a8e4e26a1f71 ✅
# b2f6663bc45d43b89781a86145fe3b1e ✅
# 42a73465fe0749d79a3321a6c67ec577 ✅
# 610a03f1d8204e14bc427609c960c400 ✅
# a47aa4c3a6724e4088e2e198bc279853 ✅
# d4578883d8764b35b00a5716b75d9d43 🔄
# 22cd5f85c4724d25a1681a4a4d00c30e 🔄
# 4f7a1fd4afab44929ddca72f03dcd61a 🔄
# d4578883d8764b35b00a5716b75d9d43 🔄
# 2a8c59f91e3c41e987211e8718a3e11e 🔄
# d0a845bb3bf942e282b31aa06e77c5bc 🔄
# 17d66b4f26324c79a0294c45290a0770 🔄
# b182a91c47e3493e85a5e5e99f9fe86a 🔄
# ab5406b3a0034740bc9261dea63281ad 🔄
# 6c6a77ea79334d36b12778f2d6a2b90f 🔄
# 00183d9d3cd64faf816fd0d38fc87b41 🔄


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_velocity(
    positions, velocity, gbest, pbest, center, magnitude, dispersion, num_particles
):
    prim = Primitives(device)  # noqa
    vel_str_torch = clean_func(VEL)
    # raise Exception(vel_str_torch)
    return eval(vel_str_torch)


def clean_func(func_str: str):
    return (
        func_str.replace("\n ", "")
        .replace("sub", "prim.sub_array")
        .replace("mul", "prim.mul_array")
        .replace("add", "prim.add_array")
        .replace("div", "prim.div_array")
        .replace("neg", "prim.neg_array")
        .replace("sqrt", "prim.sqrt_array")
        .replace("abs", "prim.abs_array")
        .replace("square", "prim.square_array")
        .replace("exp", "prim.exp_array")
        .replace("inv", "prim.inv_array")
        .replace("cos", "prim.cos_array")
        .replace("sin", "prim.sin_array")
        .replace("relu", "prim.relu_array")
        .replace("kill", "prim.kill_array")
        .replace("norm", "prim.norm_array")
        .replace("pi", "3.141592653589793")
        # .replace("tensor([", "")
        # .replace("], device='cuda:0')", "")
    )


def simplified_func(
    positions,
    velocity,
    gbest,
    pbest,
    center,
    magnitude,
    dispersion,
    num_particles,
):
    # Constants
    c1 = torch.tensor([-0.1], device=device)
    c2 = torch.tensor([0], device=device)
    pi_neg = -torch.pi

    # Common computations
    term1 = positions
    term2 = magnitude / c1
    term3 = (num_particles / center) * (positions)
    term4 = positions
    term5 = c1 * torch.cos(num_particles)
    term6 = gbest + (c1 - pi)
    term7 = pbest / (pi_neg**2)
    term8 = -num_particles / pi
    # Inner cosine expression
    inner_cos = torch.cos(term4 + term5 + term6 + term7 + term8)

    # Outer cosine multiplication
    outer_cos = torch.cos(num_particles) * (term1 - positions)

    # Compute final result
    return (positions - gbest) * c1 - (
        (torch.cos(inner_cos + outer_cos + term2 + term3)) / pi
    )


def load_best_velocity_yaml(run_id: str):
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="best_velocity_function.yaml"
    )
    with open(path, "r") as f:
        return yaml.safe_load(f)["best_velocity_function"]


# Generate the custom velocity function dynamically
def make_custom_velocity_from_yaml(run_id, device):
    formula_str = load_best_velocity_yaml(run_id)
    cleaned_formula = clean_func(formula_str)

    def custom_velocity(
        positions, velocity, gbest, pbest, center, magnitude, dispersion, num_particles
    ):
        prim = Primitives(device)  # noqa
        return eval(cleaned_formula)

    return custom_velocity
