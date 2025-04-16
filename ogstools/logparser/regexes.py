# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from dataclasses import dataclass


@dataclass
class Log:
    type: str
    line: int

    @staticmethod
    def type_str() -> str:
        return "Log"


class Info(Log):
    @staticmethod
    def type_str() -> str:
        return "Info"


class WarningType(Log):
    @staticmethod
    def type_str() -> str:
        return "Warning"


class ErrorType(Log):
    @staticmethod
    def type_str() -> str:
        return "Error"


class CriticalType(Log):
    @staticmethod
    def type_str() -> str:
        return "Critical"


@dataclass
class MPIProcess(Info):
    mpi_process: int


@dataclass
class NoRankOutput:
    pass


@dataclass
class OGSVersionLog(MPIProcess, NoRankOutput):
    version: str


@dataclass
class OGSVersionLog2(MPIProcess, NoRankOutput):
    ogs_version: str
    log_version: int = 0
    log_level: str = ""


@dataclass
class AssemblyTime(MPIProcess, Info):
    assembly_time: float


@dataclass
class TimeStepEnd(MPIProcess, Info):
    time_step: int


@dataclass
class IterationStart(MPIProcess, Info):
    iteration_number: int


@dataclass
class IterationEnd(MPIProcess, Info):
    iteration_number: int
    iteration_time: float


@dataclass
class CouplingIterationStart(MPIProcess, Info):
    coupling_iteration_number: int


@dataclass
class CouplingIterationEnd(MPIProcess, Info):
    coupling_iteration_number: int
    # coupling_iteration_time: float


@dataclass
class TimeStepStart(MPIProcess, Info):
    time_step: int
    step_start_time: float
    step_size: float


@dataclass
class TimeStepOutputTime(MPIProcess, Info):
    time_step: int
    output_time: float


@dataclass
class SolvingProcessStart(MPIProcess, Info):
    process: int


@dataclass
class TimeStepSolutionTime(MPIProcess, Info):
    process: int
    time_step_solution_time: float
    time_step: int


@dataclass
class TimeStepSolutionTimeCoupledScheme(MPIProcess, Info):
    process: int
    time_step_solution_time: float
    time_step: int
    coupling_iteration: int


@dataclass
class TimeStepFinishedTime(MPIProcess, Info):
    time_step: int
    time_step_finished_time: float


@dataclass
class DirichletTime(MPIProcess, Info):
    dirichlet_time: float


@dataclass
class LinearSolverTime(MPIProcess, Info):
    linear_solver_time: float


@dataclass
class MeshReadTime(MPIProcess, Info):
    mesh_read_time: float


@dataclass
class SimulationExecutionTime(MPIProcess, Info):
    execution_time: float


@dataclass
class ComponentConvergenceCriterion(MPIProcess, Info):
    component: int
    dx: float
    x: float
    dx_x: float


@dataclass
class TimeStepConvergenceCriterion(MPIProcess, Info):
    dx: float
    x: float
    dx_x: float


@dataclass
class CouplingIterationConvergence(MPIProcess, Info):
    coupling_iteration_process: int


@dataclass
class GenericCodePoint(MPIProcess, Info):
    message: str


@dataclass
class PhaseFieldEnergyVar(MPIProcess, Info):
    elastic_energy: float
    surface_energy: float
    pressure_work: float
    total_energy: float


@dataclass
class ErrorMessage(MPIProcess, ErrorType):
    message: str


@dataclass
class CriticalMessage(MPIProcess, CriticalType):
    message: str


@dataclass
class WarningMessage(MPIProcess, WarningType):
    message: str


@dataclass
class SimulationStartTime(MPIProcess, Info, NoRankOutput):
    message: str


@dataclass
class Termination:
    pass


@dataclass
class SimulationEndTime(MPIProcess, Info, Termination):
    message: str


def ogs_regexes() -> list[tuple[str, type[Log]]]:
    """
    Defines regular expressions for parsing OpenGeoSys log messages.

    :returns:  A list of tuples, each containing a regular expression pattern
              and the corresponding message class.
    """
    return [
        (
            r"info: This is OpenGeoSys-6 version (\d+)\.(\d+)\.(\d+)(?:-(\d+))?(?:-g([0-9a-f]+))?(?:\.dirty)?",
            OGSVersionLog,
        ),
        (
            r"info: \[time\] Output of timestep (\d+) took ([\d\.e+-]+) s",
            TimeStepOutputTime,
        ),
        (
            r"info: \[time\] Time step #(\d+) took ([\d\.e+-]+) s",
            TimeStepFinishedTime,
        ),
        (r"info: \[time\] Reading the mesh took ([\d\.e+-]+) s", MeshReadTime),
        (
            r"info: \[time\] Execution took ([\d\.e+-]+) s",
            SimulationExecutionTime,
        ),
        (
            r"info: \[time\] Solving process #(\d+) took ([\d\.e+-]+) s in time step #(\d+)  coupling iteration #(\d+)",
            TimeStepSolutionTimeCoupledScheme,
        ),
        (
            r"info: \[time\] Solving process #(\d+) took ([\d\.e+-]+) s in time step #(\d+)",
            TimeStepSolutionTime,
        ),
        (
            r"info: === Time stepping at step #(\d+) and time ([\d\.e+-]+) with step size (.*)",
            TimeStepStart,
        ),
        (r"info: \[time\] Assembly took ([\d\.e+-]+) s", AssemblyTime),
        (
            r"info: \[time\] Applying Dirichlet BCs took ([\d\.e+-]+) s",
            DirichletTime,
        ),
        (
            r"info: \[time\] Linear solver took ([\d\.e+-]+) s",
            LinearSolverTime,
        ),
        (r"info: Iteration #(\d+) started", IterationStart),
        (
            r"info: \[time\] Iteration #(\d+) took ([\d\.e+-]+) s",
            IterationEnd,
        ),
        (
            r"info: Convergence criterion: \|dx\|=([\d\.e+-]+), \|x\|=([\d\.e+-]+), \|dx\|/\|x\|=([\d\.e+-]+|nan|inf)$",
            TimeStepConvergenceCriterion,
        ),
        (
            r"info: Elastic energy: ([\d\.e+-]+) Surface energy: ([\d\.e+-]+) Pressure work: ([\d\.e+-]+) Total energy: ([\d\.e+-]+)",
            PhaseFieldEnergyVar,
        ),
        (
            r"info: ------- Checking convergence criterion for coupled solution of process #(\d+)",
            CouplingIterationConvergence,
        ),
        (
            r"info: ------- Checking convergence criterion for coupled solution  of process ID (\d+) -------",
            CouplingIterationConvergence,
        ),
        (
            r"info: Convergence criterion, component (\d+): \|dx\|=([\d\.e+-]+), \|x\|=([\d\.e+-]+), \|dx\|/\|x\|=([\d\.e+-]+|nan|inf)$",
            ComponentConvergenceCriterion,
        ),
        ("critical: (.*)", CriticalMessage),
        ("error: (.*)", ErrorMessage),
        ("warning: (.*)", WarningMessage),
        (
            r"info: OGS started on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4}).",
            SimulationStartTime,
        ),
        (
            r"info: OGS terminated on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4}).",
            SimulationEndTime,
        ),
    ]


def new_regexes() -> list[tuple[str, type[Log]]]:
    return [
        (
            r"info: This is OpenGeoSys-6 version ([\w\-\.]+)\. Log version: (\d+), Log level: (\w+)\.",
            OGSVersionLog2,
        ),
        (
            r"info: === Time stepping at step #(\d+) and time ([\d\.e+-]+) with step size (\d+)",
            TimeStepStart,
        ),
        (
            r"info: \[time\] Time step #(\d+) took",
            TimeStepEnd,
        ),
        (r"info: Iteration #(\d+) started", IterationStart),
        (
            r"info: \[time\] Iteration #(\d+) took ([\d\.e+-]+) s",
            IterationEnd,
        ),
        (r"info: Solving process #(\d+) started\.", SolvingProcessStart),
        (
            r"info: \[time\] Solving process #(\d+) took ([\d\.e+-]+) s in time step #(\d+)",
            TimeStepSolutionTime,
        ),
        (
            r"info: Global coupling iteration #(\d+) started",
            CouplingIterationStart,
        ),
        (
            r"info: \[time\] Global coupling iteration #(\d+) took",
            CouplingIterationEnd,
        ),
        (
            r"info: Simulation completed on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\.",
            SimulationEndTime,
        ),
        (
            r"error: Simulation aborted on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\.",
            SimulationEndTime,
        ),
    ]
