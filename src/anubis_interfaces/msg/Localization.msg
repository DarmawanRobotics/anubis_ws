
std_msgs/Header header       # timestamp is when the measurement was taken

uint8 STATUS_INITIALIZING = 0
uint8 STATUS_RELOCALIZING = 1
uint8 STATUS_RELOCALIZATION_SUCCEEDED = 2
uint8 STATUS_NORMAL = 3
uint8 STATUS_LOST = 4
uint8 status                 # one of the STATUS_* constants above

uint8 COORD_MAP = 0
uint8 COORD_GEODETIC = 1
uint8 coord_type             # one of the COORD_* constants above -- this
                              # project only ever uses COORD_MAP (no GPS)

geometry_msgs/Vector3 pos     # position (x, y, z) in map coordinates, meters
geometry_msgs/Vector3 rpy     # orientation (roll, pitch, yaw), radians
geometry_msgs/Vector3 vel     # linear velocity, m/s
geometry_msgs/Vector3 acc     # linear acceleration, m/s^2
geometry_msgs/Vector3 gyro    # angular velocity, rad/s
float64 speed                 # scalar speed, m/s
