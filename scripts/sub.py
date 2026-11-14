import cv2, sys
from physicalai.capture.transport import SharedCamera
for name, svc in [("top","physicalai/camera/RealSenseCamera/243622063627/frame"),("front","physicalai/camera/RealSenseCamera/243622061066/frame")]+[("wrist",s) for s in sys.argv[1:]]:
    try:
        cam = SharedCamera.from_publisher(svc, color_mode="rgb"); cam.connect(timeout=5)
        fr = cam.read(timeout=3); img = getattr(fr, "data", fr)
        img = img if hasattr(img, "shape") else fr.image
        cv2.imwrite(f"cam_{name}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); print(name, img.shape); cam.disconnect()
    except Exception as e: print(name, "ERR", type(e).__name__, e)
