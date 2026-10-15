# Idle % golden fixture — hand computation

Definition (README): idle = running AND idle_rpm_band.low <= rpm <= idle_rpm_band.high AND load < idle_load_cutoff.
Fixture config: band 600–800 rpm (inclusive), cutoff 15 % (strict), load = SPN 92. Running = rpm > 300.
Canonical rows are 1 s; every second carries constant rpm/load, so each row is exact.

| seconds | rpm | load | running | idle | why |
|---|---|---|---|---|---|
| 0–9 (10) | 0 | 0 | no | no | engine off |
| 10–28 (19) | 700 | 8 | yes | **yes** | in band, 8 < 15 |
| 29 (1) | 700 | not available | yes | no | load unknown (NA decodes to NULL, never held) |
| 30–39 (10) | 700 | 30 | yes | no | load ≥ cutoff |
| 40–49 (10) | 1500 | 10 | yes | no | rpm above band |
| 50–54 (5) | 800 | 14 | yes | **yes** | upper band edge is inclusive |
| 55–59 (5) | 750 | 15 | yes | no | 15 is not < 15 |
| 60–69 (10) | 650 | 5 | yes | **yes** | in band, 5 < 15 |

Running seconds = 19+1+10+10+5+5+10 = **60**. Idle seconds = 19+5+10 = **34**.
Idle % = 34 / 60 × 100 = **56.666… %**. Load unknown while running = 1 / 60 = 1.666… %.
The duplicated EEC1 frame at s 12 must not change anything.
