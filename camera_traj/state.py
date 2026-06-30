from geometry_msgs.msg import TransformStamped

from .missions import Mission


class State:
    params: dict = {}
    mission: Mission
    last_transform_to_car: TransformStamped = None
    last_transform_to_world: TransformStamped = None


node_state: State = State()
