# Disable torque on all 6 SO101 follower motors (serial 5B3D040988). Arm goes limp.
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
B, G = MotorNormMode.RANGE_M100_100, MotorNormMode.RANGE_0_100
names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
motors = {n: Motor(i+1, "sts3215", G if n=="gripper" else B) for i, n in enumerate(names)}
port = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040988-if00"
bus = FeetechMotorsBus(port=port, motors=motors)
bus.connect(handshake=False)
bus.disable_torque()
print({n: bus.read("Torque_Enable", n, normalize=False) for n in names})
bus.disconnect(disable_torque=False)
