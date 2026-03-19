#!/usr/bin/env python
"""Update test reference files

Maintainers run these tests updates on Linux regularly (before release).

    python scripts/update_tests.py
"""

import shutil

from ogstools.examples import (
    load_model_liquid_flow_simple,
    log_lf_simple_ranks_1,
    log_lf_simple_ranks_3,
    log_lf_simple_ranks_none,
)


def update_convergence_logs():
    """Run liquid flow simple with each rank configuration and save log files."""
    params = [
        (None, log_lf_simple_ranks_none),
        (1, log_lf_simple_ranks_1),
        (3, log_lf_simple_ranks_3),
    ]
    for ranks, log_file in params:
        model = load_model_liquid_flow_simple()
        model.execution.mpi_ranks = ranks
        sim = model.run()
        assert (
            sim.status == sim.Status.done
        ), f"Simulation failed for ranks={ranks}"
        shutil.copy(sim.log_file, log_file)
        print(f"Saved {log_file}")


if __name__ == "__main__":
    update_convergence_logs()
