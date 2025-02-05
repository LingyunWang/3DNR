# 3DNR

多帧融合用于视频降噪

使用DIS稠密光流来做块匹配

对相似块做融合，融合权重根据源patch的梯度，目标patch之间的图像相似度计算。

运行demo：
```bash
python mfdn2.py
```

参考：
《MULTI-FRAME IMAGE DENOISING AND STABILIZATION》中的权重融合方法
https://zhuanlan.zhihu.com/p/495897164