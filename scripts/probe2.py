import pyrealsense2 as rs, numpy as np
O=rs.option
p=rs.pipeline(); c=rs.config(); c.enable_device("243622063627"); c.enable_stream(rs.stream.color,1920,1080,rs.format.bgr8,30)
prof=p.start(c); sen=prof.get_device().first_color_sensor()
print([str(o) for o in sen.get_supported_options()])
for _ in range(60): f=p.wait_for_frames()
img=np.asanyarray(f.get_color_frame().get_data()); print("auto mean",img.mean(axis=(0,1)))
for o in [O.exposure,O.gain,O.white_balance,O.enable_auto_exposure,O.enable_auto_white_balance]:
    r=sen.get_option_range(o); print(o, sen.get_option(o), r.min, r.max, r.step, r.default)
p.stop()
