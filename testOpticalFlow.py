import cv2
import os
import numpy as np
import time

def flow2rgb(flow):
    h, w, c = flow.shape
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def draw_flow(img, flow, stride, scale):
    h, w, c = img.shape
    y_start = stride//2
    x_start = stride//2
    for y in range(y_start, h, stride):
        for x in range(x_start, w, stride):
            v_x, v_y = int(flow[y//scale, x//scale, 0]) * scale, int(flow[y//scale, x//scale, 1]) * scale
            img = cv2.line(img, (x, y), (x+v_x, y+v_y), (128, 187, 54), 2)
            img = cv2.circle(img, (x, y), 5, (165, 112, 78), -1)
    return img

def warp_flow(img, flow):
    h, w, c = img.shape
    
    flow_map = np.kron(flow, np.ones((8, 8, 1))) * 8
    
    flow_map[:, :, 0] += np.arange(w) 
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis] 
    flow_map = flow_map.astype(np.float32)
    
    res = np.array((h, w, c), dtype=np.float32)
    res = cv2.remap(img, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR)

    return res

if __name__ == '__main__':
    video_dir = 'E:/1080P/130'
    video_name = 'HoverX1-0296.mp4'
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))

    #save_path = './296-flow.mp4'
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    #fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    #out = cv2.VideoWriter(save_path, fourcc, frame_fps, (frame_w, frame_h), True)

    img_gray_prev = None
    img_prev = None
    img_idx = 0
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    while True:
        print(img_idx)
        ret, img = cap.read()
        if not ret:
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_gray_prev is not None:
            
            #feature_pts = cv2.goodFeaturesToTrack(img_gray_prev, maxCorners=100, qualityLevel=0.01, minDistance=10)
            #print(feature_pts.shape)
            #match_pts, st, err = cv2.calcOpticalFlowPyrLK(img_gray_prev, img_gray, feature_pts, None)
            #print(st)

            img_gray_prev_half = cv2.resize(img_gray_prev, (frame_w//8, frame_h//8))
            img_gray_half = cv2.resize(img_gray, (frame_w//8, frame_h//8))

            t = time.time()
            #flow = dis.calc(img_gray_prev_half, img_gray_half, None)
            flow = cv2.calcOpticalFlowFarneback(img_gray_prev_half, img_gray_half, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            print('optical flow cost: ', time.time()-t)

            img_draw = draw_flow(img_prev, flow, stride=32, scale=8)
            cv2.imwrite(str(img_idx-1)+'.png', img_draw)
            
            #warp_flow_img = warp_flow(img, flow)
            #cv2.imwrite(str(img_idx-1)+'warp.png', warp_flow_img)
            
            # 使用色彩编码可视化光流
            #rgb = flow2rgb(flow)
            
            #out.write(img_draw)
            
        img_gray_prev = img_gray
        img_prev = img
        img_idx += 1

        #if img_idx > 5:
        #    break

    #out.release()



      '''
    # 选择好的角点并绘制轨迹
    good_new = match_pts[st==1]
    good_old = track_pts[st==1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        a = int(a) 
        b = int(b)
        c, d = old.ravel()
        c = int(c)
        d = int(d)
        img2_gray = cv2.line(img2_gray, (a, b), (c, d), (128, 187, 54), 2)
        img2_gray = cv2.circle(img2_gray, (c, d), 5, (165, 112, 78), -1)
        img0_gray = cv2.circle(img0_gray, (a, b), 5, (165, 112, 78), -1)
    cv2.imwrite('debug2.png', img2_gray)
    cv2.imwrite('debug0.png', img0_gray)
    return 
    '''