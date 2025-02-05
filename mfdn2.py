import cv2
import os
import numpy as np
import time

def patch_distance(patch1, patch2):
    res = np.abs(np.int16(patch1) - np.int16(patch2))
    return np.sum(res)

# find most similar patch in dst_img
def match_patch2(match_pts0, st0, array_idx, patch_size, h, w):
    if st0[array_idx] == 0:
        dst_y, dst_x = -1, -1
    elif st0[array_idx] == 1:
        dst_y, dst_x = match_pts0[array_idx, 0, 1], match_pts0[array_idx, 0, 0]
        if dst_y - patch_size // 2 < 0 or dst_y + patch_size // 2 > h \
                or dst_x - patch_size // 2 < 0 or dst_x + patch_size // 2 > w:
            dst_y, dst_x = -1, -1    
        else:
            dst_y, dst_x = int(dst_y) - patch_size // 2, int(dst_x) - patch_size // 2    
    return dst_y, dst_x

# find most similar patch in dst_img
def match_patch3(flow, x, y, patch_size, scale, h, w):
    x_c = x + patch_size//2
    y_c = y + patch_size//2
    dst_x, dst_y = x + int(flow[y_c//scale, x_c//scale, 0] * scale), y + int(flow[y_c//scale, x_c//scale, 1] * scale)
    
    pixels = []
    
    for i, j in [(0, 0)]:
        px = dst_x + j
        py = dst_y + i
        if px >= 0 and px + patch_size < w \
            and py >= 0 and py + patch_size < h:
            pixels.append((px, py))
    
    return pixels

# 计算patch的相似度
def match_patch_dist(patch1, img0, dst_y, dst_x):
    patch_size = patch1.shape[0]
    dist0 = patch_distance(patch1=patch1, patch2=img0[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size])
    dist0 = float(dist0) / (patch_size * patch_size * 400)
    return dist0

# 计算patch的梯度
def patch_gradient(patch):
    patch_size = patch.shape[0]
    hor_grad_sum = np.sum(np.abs(patch[:, 1:] - patch[:, :-1]))
    ver_grad_sum = np.sum(np.abs(patch[1:, :] - patch[:-1, :]))
    return float(hor_grad_sum + ver_grad_sum) / (patch_size - 1) / patch_size

# 辅助函数， 画出光流
def draw_flow(img_in, flow, stride, scale):
    img = img_in.copy()
    h, w, c = img.shape
    y_start = stride//2
    x_start = stride//2
    for y in range(y_start, h, stride):
        for x in range(x_start, w, stride):
            v_x, v_y = int(flow[y//scale, x//scale, 0]) * scale, int(flow[y//scale, x//scale, 1]) * scale
            img = cv2.line(img, (x, y), (x+v_x, y+v_y), (128, 187, 54), 2)
            img = cv2.circle(img, (x, y), 5, (165, 112, 78), -1)
    return img

# 通过光流进行图像重构， 通过双线性插值 缺点是，大尺度图像太耗时，尤其在移动端。
def warp_flow(img, flow):
    h, w, c = img.shape
    
    flow_map = np.kron(flow, np.ones((8, 8, 1))) * 8
    
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis] 
    flow_map = flow_map.astype(np.float32)
    
    res = np.array((h, w, c), dtype=np.float32)
    res = cv2.remap(img, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR)

    return res

# 通过稠密光流，找到目标帧中和源帧最相似的patch。通过多帧融合去噪。融合权重来自于patch的梯度和patch间的相似度。
def mfdn_dense(imgs, gray_imgs, flow_queue, patch_size, img_idx):
    img0, img1, img2, img3, img4 = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]
    
    img0_gray, img1_gray, img2_gray, img3_gray, img4_gray = gray_imgs[0], gray_imgs[1], gray_imgs[2], gray_imgs[3], gray_imgs[4]

    assert len(flow_queue) == 4
    flow0, flow1, flow3, flow4 = flow_queue[0], flow_queue[1], flow_queue[2], flow_queue[3]

    h, w, c = img0.shape

    #draw0 = draw_flow(img2, flow0, patch_size, 8)
    #draw1 = draw_flow(img2, flow1, patch_size, 8)
    #draw3 = draw_flow(img2, flow3, patch_size, 8)
    #draw4 = draw_flow(img2, flow4, patch_size, 8)

    thes = 0.5 * patch_size
    v_count = (h//patch_size)
    h_count = (w//patch_size)
    t = time.time()
    for vi in range(v_count):
        for hi in range(h_count):
            # patch left
            x = hi * patch_size
            y = vi * patch_size  
            
            # filter formative movement patch
            vx0 = np.abs(flow0[y//8, x//8, 0] * 8)
            vy0 = np.abs(flow0[y//8, x//8, 1] * 8)
            vx1 = np.abs(flow1[y//8, x//8, 0] * 8)
            vy1 = np.abs(flow1[y//8, x//8, 1] * 8)
            vx3 = np.abs(flow3[y//8, x//8, 0] * 8)
            vy3 = np.abs(flow3[y//8, x//8, 1] * 8)
            vx4 = np.abs(flow4[y//8, x//8, 0] * 8)
            vy4 = np.abs(flow4[y//8, x//8, 1] * 8)
            
            #if (vx0 + vy0) > thes or (vx4 + vy4) > thes or (vx1 + vy1) > thes or (vx3 + vy3) > thes:
                #debug
                #img2[y:y+patch_size, x:x+patch_size] = np.int8(img2[y:y+patch_size, x:x+patch_size] * 0.7 + 72)
                #continue

            # patch
            patch2 = img2[y:y+patch_size, x:x+patch_size]
            patch2_gray = img2_gray[y:y+patch_size, x:x+patch_size]
            gradient = patch_gradient(patch=patch2_gray)
            
            patch_sum = img2[y:y+patch_size, x:x+patch_size].astype(np.float32)
            w_sum = 1.0
            pixels0 = match_patch3(flow0, x, y, patch_size, scale=8, h=h, w=w)
            for i in range(len(pixels0)):
                py, px = (pixels0[i][1], pixels0[i][0])
                dist0 = match_patch_dist(patch2, img0, py, px)
                w0 = np.exp(-gradient * 0.1 * dist0)
                w_sum += w0
                patch_sum += w0 * img0[py:py + patch_size, px:px + patch_size]


            pixels1 = match_patch3(flow1, x, y, patch_size, scale=8, h=h, w=w)
            for i in range(len(pixels1)):
                py, px = (pixels1[i][1], pixels1[i][0])
                dist1 = match_patch_dist(patch2, img1, py, px)
                w1 = np.exp(-gradient * 0.1 * dist1)
                w_sum += w1
                patch_sum += w1 * img1[py:py + patch_size, px:px + patch_size]
        

            pixels3 = match_patch3(flow3, x, y, patch_size, scale=8, h=h, w=w)
            for i in range(len(pixels3)):
                py, px = (pixels3[i][1], pixels3[i][0])
                dist3 = match_patch_dist(patch2, img3, py, px)
                w3 = np.exp(-gradient * 0.1 * dist3)
                w_sum += w3
                patch_sum += w3 * img3[py:py + patch_size, px:px + patch_size]
                

            pixels4 = match_patch3(flow4, x, y, patch_size, scale=8, h=h, w=w)
            for i in range(len(pixels4)):
                py, px = (pixels4[i][1], pixels4[i][0])
                dist4 = match_patch_dist(patch2, img4, py, px)
                w4 = np.exp(-gradient * 0.1 * dist4)
                w_sum += w4
                patch_sum += w4 * img4[py:py + patch_size, px:px + patch_size]
                
            img2[y:y+patch_size, x:x+patch_size] = np.int8(patch_sum / w_sum)
    
    print('time:', time.time() - t)
    
    return img2 #, draw0 , draw1, draw3, draw4


if __name__ == '__main__':
    video_dir = 'noise'
    video_name = 'test.mp4'
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))

    save_path = './test-dn.mp4'
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(save_path, fourcc, frame_fps, (frame_w, frame_h), True)
    #out0 = cv2.VideoWriter('flow0.mp4', fourcc, frame_fps, (frame_w, frame_h), True)
    #out1 = cv2.VideoWriter('flow1.mp4', fourcc, frame_fps, (frame_w, frame_h), True)
    #out3 = cv2.VideoWriter('flow3.mp4', fourcc, frame_fps, (frame_w, frame_h), True)
    #out4 = cv2.VideoWriter('flow4.mp4', fourcc, frame_fps, (frame_w, frame_h), True)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    img_idx = 0
    img_queue = []
    img_gray_queue = []
    img_gray_s_queue = []
    flow_queue = []
    img_gray_prev = None
    while True:
        print(img_idx)
        ret, img = cap.read()
        if not ret:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_s = cv2.resize(img_gray, (frame_w//8, frame_h//8))

        if len(img_queue) < 5:
            img_queue.append(img)
            img_gray_queue.append(img_gray)
            img_gray_s_queue.append(img_gray_s)
        else:
            img_queue.pop(0)
            img_queue.append(img)
            img_gray_queue.pop(0)
            img_gray_queue.append(img_gray)
            img_gray_s_queue.pop(0)
            img_gray_s_queue.append(img_gray_s)

        if len(img_queue) == 5:
            flow0 = dis.calc(img_gray_s_queue[2], img_gray_s_queue[0], None)
            flow1 = dis.calc(img_gray_s_queue[2], img_gray_s_queue[1], None)
            flow3 = dis.calc(img_gray_s_queue[2], img_gray_s_queue[3], None)
            flow4 = dis.calc(img_gray_s_queue[2], img_gray_s_queue[4], None)
            #flow0 = cv2.calcOpticalFlowFarneback(img_gray_s_queue[2], img_gray_s_queue[0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #flow1 = cv2.calcOpticalFlowFarneback(img_gray_s_queue[2], img_gray_s_queue[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #flow3 = cv2.calcOpticalFlowFarneback(img_gray_s_queue[2], img_gray_s_queue[3], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #flow4 = cv2.calcOpticalFlowFarneback(img_gray_s_queue[2], img_gray_s_queue[4], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_queue = [flow0, flow1, flow3, flow4]
            
        
        if len(img_queue) == 5:
            #, flow0, flow1, flow3, flow4
            img_fusion  = mfdn_dense(img_queue, img_gray_queue, flow_queue, 30, img_idx)   #, d0, d1, d3, d4
            out.write(img_fusion)
            #out0.write(flow0)
            #out1.write(flow1)
            #out3.write(flow3)
            #out4.write(flow4)
        
        img_idx += 1
        
    out.release()


  