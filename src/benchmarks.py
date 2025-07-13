from typing import Callable, Literal
import torch


BENCHMARK_NAMES = [
    "ackley",
    "sphere",
    "rastrigin",
    "cigar",
    "rosenbrock",
    "easom",
    "griewank",
    "schwefel",
    "levy",
    "zakharov",
]

PARTITIONS = {
    1: {
        "train": [
            "ackley",
            "sphere",
            "rastrigin",
            "easom",
            "griewank",
        ],
        "test": [
            "cigar",
            "rosenbrock",
            "schwefel",
            "levy",
            "zakharov",
        ],
    },
    2: {
        "train": [
            "cigar",
            "rosenbrock",
            "schwefel",
            "levy",
            "zakharov",
        ],
        "test": [
            "ackley",
            "sphere",
            "rastrigin",
            "easom",
            "griewank",
        ],
    },
    3: {
        "train": [
            "ackley",
            "griewank",
            "rastrigin",
            "zakharov",
            "rosenbrock",
        ],
        "test": [
            "sphere",
            "easom",
            "cigar",
            "levy",
            "schwefel",
        ],
    },
    4: {
        "train": [
            "sphere",
            "levy",
            "cigar",
            "easom",
            "ackley",
        ],
        "test": [
            "rosenbrock",
            "griewank",
            "schwefel",
            "zakharov",
            "rastrigin",
        ],
    },
    5: {
        "train": [
            "schwefel",
            "rastrigin",
            "griewank",
            "easom",
            "zakharov",
        ],
        "test": [
            "cigar",
            "rosenbrock",
            "ackley",
            "levy",
            "sphere",
        ],
    },
}


def get_benchmarks_partitions():
    benchmark_partition = {
        name: partition["test"] for name, partition in PARTITIONS.items()
    }
    # create a dictionary as such that "cigar": [1, 3, 5], etc, given that we have {1: ['cigar', ...], 2: [...], ...}
    partition_benchmark = {}
    for partition, benchmarks in benchmark_partition.items():
        for benchmark in benchmarks:
            if benchmark in partition_benchmark:
                partition_benchmark[benchmark].append(partition)
            else:
                partition_benchmark[benchmark] = [partition]

    # now do it with sets, whilst simplifying the logic
    partition_benchmark = {}
    for partition, benchmarks in benchmark_partition.items():
        for benchmark in benchmarks:
            partition_benchmark.setdefault(benchmark, set()).add(partition)

    # make the sets into lists
    partition_benchmark = {k: list(v) for k, v in partition_benchmark.items()}
    return partition_benchmark


class BenchmarkFunction(Callable):
    def __init__(
        self,
        benchmark_name: Literal[
            "ackley",
            "sphere",
            "rastrigin",
            "cigar",
            "rosenbrock",
            "easom",
            "griewank",
            "schwefel",
            "levy",
            "zakharov",
        ],
        dimensions: int,
        bounds: tuple[float, float],
        num_problems: int,
        device: torch.device,
        optimum_bounds: tuple[float, float],
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.name = benchmark_name
        self.function = getattr(self, benchmark_name)
        self.dimensions = dimensions
        self.bounds = bounds
        self.optimum_bounds = optimum_bounds
        assert num_problems >= 1 and isinstance(num_problems, int), (
            f"Number of problems must be an integer greater than 0. Current value: {num_problems}"
        )
        self.num_problems = num_problems
        self.device = device
        self.optimum = (
            torch.rand(num_problems, dimensions, device=device)
            * (optimum_bounds[1] - optimum_bounds[0])
            + optimum_bounds[0]
        )
        self._optimum = self.optimum.unsqueeze(dim=1).to(self.device)
        self._rotation = self._generate_random_rotation_matrix()
        assert self.optimum.shape == (num_problems, dimensions), (
            f"Optimum shape must be ({num_problems}, {dimensions}). "
            f"Current shape: {self.optimum.shape}"
        )

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        shifted_positions = positions - self._optimum
        rotated_positions = torch.matmul(
            shifted_positions, self._rotation.transpose(-1, -2)
        )
        return self.function(rotated_positions)

    def __str__(self):
        return self.name

    @staticmethod
    def ackley(swarms: torch.Tensor) -> torch.Tensor:
        a, b, c = 20, 0.2, 2 * torch.pi
        dims = swarms.shape[-1]
        sum_sq_term = torch.sum(swarms**2, dim=2) / dims
        cos_term = torch.sum(torch.cos(c * swarms), dim=2) / dims

        result = (
            -a * torch.exp(-b * torch.sqrt(sum_sq_term))
            - torch.exp(cos_term)
            + a
            + torch.exp(torch.tensor(1.0))
        )
        return result

    @staticmethod
    def cigar(swarms: torch.Tensor) -> torch.Tensor:
        x1_squared = swarms[:, :, 0] ** 2
        other_dims = torch.sum(swarms[:, :, 1:] ** 2, dim=2)
        result = x1_squared + 10**6 * other_dims
        return result

    @staticmethod
    def easom(swarms: torch.Tensor) -> torch.Tensor:
        """
        Easom function for multiple dimensions.
        The global minimum is at (pi, pi, ..., pi) with a value of -1.
        """
        cos_term = torch.prod(torch.cos(swarms), dim=2)
        exp_term = torch.exp(-torch.sum((swarms) ** 2, dim=2))
        result = -cos_term * exp_term
        return result

    @staticmethod
    def sphere(swarms: torch.Tensor) -> torch.Tensor:
        result = torch.sum(swarms**2, dim=2)
        return result

    @staticmethod
    def rastrigin(swarms: torch.Tensor, a: int = 10) -> torch.Tensor:
        """
        Rastrigin function for multiple dimensions.
        """
        dimensions = swarms.shape[-1]
        sum = torch.sum(swarms**2 - a * torch.cos(2 * torch.pi * swarms), dim=-1)
        result = a * dimensions + sum
        return result

    @staticmethod
    def rosenbrock(swarms: torch.Tensor) -> torch.Tensor:
        x_i = swarms[..., :-1]
        x_next = swarms[..., 1:]
        term1 = 100 * (x_next - (x_i**2)) ** 2
        term2 = x_i**2
        result = torch.sum(term1 + term2, dim=2)
        return result

    @staticmethod
    def griewank(swarms: torch.Tensor) -> torch.Tensor:
        dimensions = swarms.shape[-1]
        sum_term = torch.sum(swarms**2, dim=2) / 4000
        indices = torch.arange(
            1, dimensions + 1, device=swarms.device, dtype=swarms.dtype
        )
        cos_term = torch.prod(torch.cos(swarms / torch.sqrt(indices)), dim=2)

        result = 1 + sum_term - cos_term
        return result

    @staticmethod
    def schwefel(swarms: torch.Tensor) -> torch.Tensor:
        dimensions = swarms.shape[-1]
        shifted_swarms = swarms + 420.968746
        sum_term = torch.sum(
            shifted_swarms * torch.sin(torch.sqrt(torch.abs(shifted_swarms))), dim=-1
        )
        result = 418.9829 * dimensions - sum_term
        return result

    @staticmethod
    def levy(swarms: torch.Tensor) -> torch.Tensor:
        w = 1 + (swarms) / 4

        term1 = torch.sin(torch.pi * w[..., 0]) ** 2
        term2 = torch.sum(
            (w[..., :-1] - 1) ** 2
            * (1 + 10 * torch.sin(torch.pi * w[..., :-1] + 1) ** 2),
            dim=2,
        )
        term3 = (w[..., -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[..., -1]) ** 2)

        result = term1 + term2 + term3
        return result

    @staticmethod
    def zakharov(swarms: torch.Tensor) -> torch.Tensor:
        D = swarms.shape[2]

        sum1 = torch.sum(swarms**2, dim=2)

        indices = torch.arange(1, D + 1, device=swarms.device, dtype=swarms.dtype)
        sum2 = torch.sum(0.5 * indices * swarms, dim=2)

        result = sum1 + sum2**2 + sum2**4
        return result

    def _generate_random_rotation_matrix(self) -> torch.Tensor:
        """
        Generates a random rotation matrix of size (num_problems, dimensions, dimensions).
        Ensures the matrix is orthogonal and has determinant +1.

        Returns:
        - A random rotation matrix for each problem.
        """
        # Generate a random matrix with normally distributed values
        random_matrix = torch.randn(
            self.num_problems, self.dimensions, self.dimensions, device=self.device
        )

        # Use QR decomposition to get an orthogonal matrix
        Q, _ = torch.linalg.qr(random_matrix)

        # Ensure the determinant is +1 (valid rotation matrix)
        # Flip the sign of the last column if determinant is -1
        det = torch.det(Q)
        Q[:, :, -1] *= torch.sign(det).unsqueeze(1)

        return Q

def generate_benchmark_instances(
    train_benchmarks: list[str],
    n_problems: int,
    dim: int,
    lower_bound: float,
    upper_bound: float,
    lower_optimum_bound: float,
    upper_optimum_bound: float,
    seed: int | None,
    device: str,
) -> list[BenchmarkFunction]:
    return [
        BenchmarkFunction(
            benchmark_name=benchmark_name,
            dimensions=dim,
            bounds=(lower_bound, upper_bound),
            num_problems=n_problems,
            device=device,
            optimum_bounds=(lower_optimum_bound, upper_optimum_bound),
            seed=seed,
        )
        for benchmark_name in train_benchmarks
    ]
