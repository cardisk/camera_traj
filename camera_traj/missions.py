from enum import IntEnum
from dataclasses import dataclass

class Mission(IntEnum):
    Manual       = 0
    Acceleration = 1
    Skidpad      = 2
    Trackdrive   = 3
    EBSTest      = 4
    Inspection   = 5
    Autocross    = 6

@dataclass
class MissionConfig:
    lap_target: int
    lap_cooldown_time_sec: float


def get_config_for_mission(mission: Mission):
    pass
