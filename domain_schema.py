# domain_schema.py
# Domain schema for schema-guided executor.

OBJECT_TYPES = {
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
    "thing", "unknown",
}
ANY_OBJECT_TYPES = {"thing", "unknown", "any", ""}
ATTRIBUTES = {
    "moving", "parked", "stopped", "standing", "not_standing",
    "stationary", "with_rider", "without_rider", "unknown",
}
RELATIONS = {
    "front", "back", "left", "right",
    "front-left", "front-right", "back-left", "back-right",
    "near", "same-status-as", "same-category-as",
}
SPATIAL_RELATIONS = {
    "front", "back", "left", "right",
    "front-left", "front-right", "back-left", "back-right", "near",
}
COMPARISON_RELATIONS = {"same-status-as", "same-category-as"}
QUESTION_TO_OPERATION = {
    "count": "count",
    "exist": "exist",
    "object": "lookup_category",
    "status": "lookup_status",
    "comparison": "compare",
}
CATEGORY_ALIASES = {
    "trafficcone": "traffic_cone",
    "traffic cone": "traffic_cone",
    "cone": "traffic_cone",
    "cones": "traffic_cone",
    "person": "pedestrian",
    "people": "pedestrian",
    "human": "pedestrian",
    "adult": "pedestrian",
    "child": "pedestrian",
    "vehicle": "car",
    "vehicles": "car",
    "construction": "construction_vehicle",
    "construction vehicle": "construction_vehicle",
}
ATTRIBUTE_ALIASES = {
    "move": "moving",
    "driving": "moving",
    "walking": "moving",
    "running": "moving",
    "park": "parked",
    "parking": "parked",
    "stop": "stopped",
    "stationary": "stopped",
    "still": "stopped",
    "standing still": "standing",
    "not standing": "not_standing",
    "sitting": "not_standing",
    "lying": "not_standing",
    "with rider": "with_rider",
    "without rider": "without_rider",
}
RELATION_ALIASES = {
    "front_left": "front-left",
    "front left": "front-left",
    "left front": "front-left",
    "front_right": "front-right",
    "front right": "front-right",
    "right front": "front-right",
    "back_left": "back-left",
    "back left": "back-left",
    "rear_left": "back-left",
    "rear left": "back-left",
    "left back": "back-left",
    "back_right": "back-right",
    "back right": "back-right",
    "rear_right": "back-right",
    "rear right": "back-right",
    "right back": "back-right",
    "behind": "back",
    "rear": "back",
    "same status": "same-status-as",
    "same_status": "same-status-as",
    "same category": "same-category-as",
    "same_category": "same-category-as",
}
