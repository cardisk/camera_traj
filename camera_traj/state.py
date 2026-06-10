from geometry_msgs.msg import TransformStamped


class State:
    params: dict
    last_transform_to_car: TransformStamped
    last_transform_to_world: TransformStamped


node_state: State = State()
