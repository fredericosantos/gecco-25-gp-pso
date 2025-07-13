from typing import Callable
import torch
from torch import Tensor

from src.benchmarks import BenchmarkFunction


class SwarmBatch:
    def __init__(
        self,
        benchmark_func: BenchmarkFunction,
        device: torch.device,
        velocity_function: Callable | None = None,
        num_particles: int = 30,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        seed: int | None = None,
        save_history: bool = False,
        vel_func_str: str | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.benchmark = benchmark_func
        self.num_problems = self.benchmark.num_problems
        self.problem_indices = torch.arange(self.num_problems, device=device)
        self.dimensions = self.benchmark.dimensions
        self.bounds = self.benchmark.bounds
        self.min_clamping = min(self.benchmark.optimum_bounds[0], self.bounds[0])
        self.max_clamping = max(self.benchmark.optimum_bounds[1], self.bounds[1])
        self.num_particles = num_particles
        self.inertia = torch.tensor(inertia)
        self.cognitive = torch.tensor(cognitive)
        self.social = torch.tensor(social)
        self.device = device
        self.save_history = save_history
        self.vel_func_str = vel_func_str

        # Initialize particle positions and velocities
        self.positions = (
            torch.rand(
                self.num_problems, num_particles, self.dimensions, device=self.device
            )
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )
        self.center_mass = self._calculate_center_mass()
        self.magnitude = self._calculate_magnitude(self.center_mass)
        self.dispersion = self._calculate_dispersion(self.magnitude)
        self.velocities = torch.rand_like(self.positions)

        # Initialize personal best positions and their fitness values
        self.pbest_positions = self.positions.clone()
        self.pbest_scores = self.benchmark(self.positions)

        # Initialize global best
        self.idxs_best = torch.argmin(self.pbest_scores, dim=1)
        pbest_positions = self.pbest_positions[self.problem_indices, self.idxs_best]
        self.gbest_positions = pbest_positions.clone()

        pbest_scores = self.pbest_scores[self.problem_indices, self.idxs_best]
        self.gbest_scores = pbest_scores.clone()

        # Track positions for animation
        self.history = []
        self._check_n_save_history()

        # Custom velocity update function
        self._custom_update_velocity = velocity_function

    def _default_update_velocity(self) -> Tensor:
        r1, r2 = torch.rand_like(self.positions), torch.rand_like(self.positions)
        inertia_ = self.inertia * self.velocities
        pbest_ = self.pbest_positions - self.positions
        cognitive_ = self.cognitive * r1 * pbest_
        gbest_ = self.gbest_positions.unsqueeze(dim=1) - self.positions
        social_ = self.social * r2 * gbest_
        self.velocities = inertia_ + cognitive_ + social_
        return self.velocities

    def _calculate_center_mass(self) -> Tensor:
        mean = torch.mean(self.positions, dim=1, keepdim=True)
        return mean

    def _calculate_magnitude(self, center_mass: Tensor) -> Tensor:
        pos_center = self.positions - center_mass
        norm = torch.linalg.norm(pos_center, dim=2, keepdim=True)
        return norm

    def _calculate_dispersion(self, magnitude: Tensor) -> Tensor:
        dispersion = torch.mean(magnitude, dim=1, keepdim=True)
        return dispersion

    def update_velocity(self) -> Tensor:
        gbest_positions = self.gbest_positions.unsqueeze(dim=1)
        gbest_positions = gbest_positions.expand_as(self.positions)
        center_mass = self.center_mass.expand_as(self.positions)
        magnitude = self.magnitude.expand_as(self.positions)
        dispersion = self.dispersion.expand_as(self.positions)
        velocities = self._custom_update_velocity(
            self.positions,
            self.velocities,
            gbest_positions,
            self.pbest_positions,
            center_mass,
            magnitude,
            dispersion,
            # torch.pi,
            torch.tensor(self.num_particles, dtype=torch.float, device=self.device),
        )
        # clamp velocities to avoid NaN values
        # velocities.clamp_(-5, 5)
        # fill NaN values with 0
        # velocities[velocities.isnan()] = 0.0

        return velocities

    def step(self):
        # Compute fitness of current positions
        fitness = self.benchmark(self.positions)

        # Update personal bests
        mask = fitness < self.pbest_scores
        mask_improved = mask.any(dim=1)
        if mask_improved.any():
            self.pbest_positions[mask] = self.positions[mask]
            self.pbest_scores[mask] = fitness[mask]

            idxs_improved = mask_improved.nonzero(as_tuple=True)[0]
            self.idxs_best[idxs_improved] = torch.argmin(
                self.pbest_scores[idxs_improved], dim=1
            )

            # Update global best
            pbest_scores = self.pbest_scores[self.problem_indices, self.idxs_best]
            pbest_positions = self.pbest_positions[self.problem_indices, self.idxs_best]

            best_mask = pbest_scores < self.gbest_scores
            self.gbest_scores[best_mask] = pbest_scores[best_mask]
            self.gbest_positions[best_mask] = pbest_positions[best_mask].clone()
        # Update velocity and positions
        if self._custom_update_velocity is None:
            self.velocities = self._default_update_velocity()
        else:
            self.velocities = self.update_velocity()
        self.positions.add_(self.velocities)

        # Ensure particles remain within bounds
        self.positions.clamp_(self.min_clamping, self.max_clamping)

        self.center_mass = self._calculate_center_mass()
        self.magnitude = self._calculate_magnitude(self.center_mass)
        self.dispersion = self._calculate_dispersion(self.magnitude)

        # Store current positions for animation as tensor
        self._check_n_save_history()

    def optimize(self, num_iterations: int, use_tqdm: bool = False):
        if use_tqdm:
            from tqdm import tqdm

            iterator = tqdm(range(num_iterations))
        else:
            iterator = range(num_iterations)
        for i in iterator:
            self.step()

        # turn self.history to list of numpy arrays
        self._check_n_save_history(to_numpy=True)

    def _check_n_save_history(self, to_numpy: bool = False):
        if self.save_history:
            if to_numpy:
                self.history = [pos.cpu().numpy() for pos in self.history]
            else:
                self.history.append(self.positions.clone())
