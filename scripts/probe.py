import cv2, pyrealsense2 as rs, numpy as np
ctx=rs.context()
for d in ctx.query_devices():
    s=d.get_info(rs.camera_info.serial_number)
    for sen in d.query_sensors():
        if sen.get_info(rs.camera_info.name)=="RGB Camera":
            print(s, "RGB opts:", [str(o) for o in sen.get_supported_options() if "exposure" in str(o) or "white" in str(o) or "gain" in str(o)])
cap=cv2.VideoCapture(16, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
for w,h in [(3840,2160),(2592,1944),(1920,1080),(1280,720)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    ok,f=cap.read(); print("wrist req",w,h,"->",ok, None if f is None else f.shape, cap.get(cv2.CAP_PROP_FPS))
cap.release()
p=rs.pipeline(); c=rs.config(); c.enable_device("243622063627"); c.enable_stream(rs.stream.color,1920,1080,rs.format.bgr8,30)
p.start(c)
for _ in range(30): f=p.wait_for_frames()
cf=f.get_color_frame()
for m in ["actual_exposure","white_balance","gain_level","auto_exposure"]:
    mv=getattr(rs.frame_metadata_value,m)
    print(m, cf.get_frame_metadata(mv) if cf.supports_frame_metadata(mv) else "unsupported")
p.stop()
