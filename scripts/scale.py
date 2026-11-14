import pyrealsense2 as rs, numpy as np, cv2
s="243622063627"
p=rs.pipeline(); c=rs.config(); c.enable_device(s); c.enable_stream(rs.stream.color,1920,1080,rs.format.bgr8,30)
prof=p.start(c)
for _ in range(60): f=p.wait_for_frames()
img=np.asanyarray(f.get_color_frame().get_data())
intr=prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
p.stop()
cv2.imwrite("top_1080.jpg", img)
print("intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f"%(intr.fx,intr.fy,intr.ppx,intr.ppy))
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# saturated blue, small blob (brick), exclude big plate
m=cv2.inRange(hsv,(100,150,80),(125,255,255))
m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
n,lab,st,cen=cv2.connectedComponentsWithStats(m)
cands=[(st[i],cen[i]) for i in range(1,n) if 2000<st[i][4]<40000]
for st_,c_ in cands:
    x,y,w,h,a=st_
    cnt=cv2.findContours((lab[y:y+h,x:x+w]>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    rect=cv2.minAreaRect(max(cnt,key=cv2.contourArea))
    print("blob at",c_.round(),"bbox",w,h,"area",a,"minAreaRect",[round(v,1) for v in rect[1]],"angle",round(rect[2],1))
    crop=img[max(0,y-40):y+h+40, max(0,x-40):x+w+40]; cv2.imwrite("brick_crop_1080.jpg",crop)
    vis=img.copy(); cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),3); cv2.imwrite("top_1080_marked.jpg",cv2.resize(vis,(960,540)))
