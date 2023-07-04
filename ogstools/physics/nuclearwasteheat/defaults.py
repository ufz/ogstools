"""Default nuclear waste models and repository setups. references:

GRS 281 Endlagerauslegung und -optimierung: Bericht zum Arbeitspaket 6
Vorläufige Sicherheitsanalyse für den Standort Gorleben p.254
https://docplayer.org/72965403-Endlagerauslegung-und-optimierung.html

**2016 models**

Projekt Ansicht: Systemanalyse für die Endlagerstandortmodelle Methode und
exemplarische Berechnungen zum Sicherheitsnachweis (2016)
https://www.bge-technology.de/fileadmin/user_upload/FuE_Berichte/Ansicht/02_ANSICHT_Systemanalyse_fuer_die_Endlagerstandortmodelle.pdf p. 27

**2020 models**

GRS 571 RESUS: Grundlagen zur Bewertung eines Endlagersystems in einer
Tongesteinsformation größerer Mächtigkeit
https://www.grs.de/sites/default/files/publications/grs-571.pdf p.49
"""  # noqa: E501,RUF100

from dataclasses import replace

import numpy as np

from ._unitsetup import Q_
from .nuclearwaste import NuclearWaste, Repository

dwr_UO2_2012 = NuclearWaste(
    name="DWR-UO2 (2012)",
    nuclide_powers=Q_([3364, 1046, 148.9, 14.41, 0.7043], "W"),
    decay_consts=np.log(2) / Q_([2.1, 32.32, 395.1, 1.446e4, 7.997e5], "yr"),
    num_bundles=12450,
    time_interim=Q_(58, "yr"),
    time_deposit=Q_(21, "yr"),
)

wwer_2012 = NuclearWaste(
    name="WWER(KGR) (2012)",
    nuclide_powers=Q_([398, 112.6, 20.34, 2.402, 8.242e-5], "W"),
    decay_consts=np.log(2) / Q_([1.769, 32.42, 430.6, 1.701e4, 1.09e6], "yr"),
    factor=2.5,
    num_bundles=5050,
    time_interim=Q_(56, "yr"),
    time_deposit=Q_(2, "yr"),
)

csd_2012 = NuclearWaste(
    name="CSD-V (2012)",
    nuclide_powers=Q_([1469, 45.49, 1.101, 0.1074], "W"),
    decay_consts=np.log(2) / Q_([27.82, 420.4, 1.237e4, 1.973e8], "yr"),
    num_bundles=3735,
    time_interim=Q_(56, "yr"),
    time_deposit=Q_(5, "yr"),
)

dwr_mix8911_2020 = NuclearWaste(
    name="DWR-mix 89/11 (2020)",
    nuclide_powers=Q_([1156, 226.7, 21.51, 0.9466], "W"),
    decay_consts=np.log(2) / Q_([32.2, 396.8, 13670, 7.593e5], "yr"),
    num_bundles=13980,
    time_interim=Q_(57, "yr"),
    time_deposit=Q_(30, "yr"),
)

wwer_2020 = NuclearWaste(
    name="WWER(KGR) (2020)",
    nuclide_powers=Q_([112.6, 20.34, 2.402, 8.243e-5], "W"),
    decay_consts=np.log(2) / Q_([32.42, 430.6, 17010, 1.09e6], "yr"),
    factor=2.5,
    num_bundles=12450,
    time_interim=Q_(57, "yr"),
    time_deposit=Q_(30, "yr"),
)

csd_2020 = NuclearWaste(
    name="CSD-V (2020)",
    nuclide_powers=Q_([1480, 44.68, 0.9507, 0.1289], "W"),
    decay_consts=np.log(2) / Q_([27.99, 417.2, 9649, 2.952e14], "yr"),
    num_bundles=12450,
    time_interim=Q_(53, "yr"),
    time_deposit=Q_(30, "yr"),
)


rk_be_2016 = NuclearWaste(
    name="RK-BE (2016)",
    nuclide_powers=Q_([42363.74, 8308.18, 3895.17, 1269.66, 842.65], "W"),
    decay_consts=Q_([262, 104, 9.12, 2.95, 0.358], "1/s") * 1e-10,
    factor=(2.0 / 3.0) * 0.72,
    num_bundles=10600,
    time_interim=Q_(23, "yr"),
    time_deposit=Q_(80, "yr"),
)

rk_ha_2016 = NuclearWaste(
    name="RK-HA (2016)",
    nuclide_powers=Q_([7583.16, 2412.91, 2458.56, 2546.25, 7231.62], "W"),
    decay_consts=Q_([82.8, 6.16, 7.15, 7.15, 82.8], "1/s") * 1e-10,
    factor=2 * 0.182,
    num_bundles=1865,
    time_interim=Q_(30, "yr"),
    time_deposit=Q_(80, "yr"),
)


repo_be_ha_2016 = Repository([rk_be_2016, rk_ha_2016])
repo_2020 = Repository([dwr_mix8911_2020, wwer_2020, csd_2020])
repo_2020_conservative = Repository(
    [replace(dwr_mix8911_2020, num_bundles=34630)]
)

waste_types = [
    dwr_UO2_2012,
    wwer_2012,
    csd_2012,
    dwr_mix8911_2020,
    wwer_2020,
    csd_2020,
    rk_be_2016,
    rk_ha_2016,
]
