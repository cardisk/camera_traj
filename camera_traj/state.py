from geometry_msgs.msg import TransformStamped


class State:
    params: dict = {}
    last_transform_to_car: TransformStamped = None
    last_transform_to_world: TransformStamped = None


node_state: State = State()
