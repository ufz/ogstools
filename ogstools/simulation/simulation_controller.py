# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


import signal
from time import sleep  # For simulation pause / interrupt
from typing import Any

from ogs.OGSSimulator import OGSSimulation


class SimulationController(OGSSimulation):
    def __init__(self, args: list[str]) -> None:
        super().__init__(args)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
        self._interrupted = False

    def _handler(self, signum: int, _: Any) -> None:
        self._interrupted = True
        print(f"Received signal {signum}, stopping...")

    def is_interrupted(self) -> bool:
        interrupted = self._interrupted
        self._interrupted = False
        return interrupted

    def wait(self, close: bool = False) -> None:
        """
        Wait until the simulation is finished.

        """
        while (
            self.current_time() < self.end_time() and not self.is_interrupted()
        ):
            self.execute_time_step()
            sleep(0.01)  # Must have, if we want to pause the simulation
        if close:
            self.close()
