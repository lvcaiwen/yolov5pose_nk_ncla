# import os
#
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# # 1 读取图像
# img = cv.imread('data/images/bus.jpg')
# cv.circle(img, (int(50), int(50)), 10, (255, 0, 0), -1)
#
#
#
# # 2 显示图像
# # 2.1 利用opencv展示图像
# cv.imshow('image',img)
# # 2.2 在matplotplotlib中展示图像
# # plt.imshow(img[:,:,::-1])
# # plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
# # plt.show()
# k = cv.waitKey(0)
# # 3 保存图像
# os.makedirs('resules',exist_ok=True)
# cv.imwrite('resules/bus.png',img)

x=[5, 5, 1, 25, 8, 4, 8, 4, 3, 5, 6, 2, 8, 4, 8, 41, 1, 87, 5, 3]
z = [5,6,2,8,4,8,41,1,87,5,3]
txt=''
for i in x+z:
    txt+=str(i)+'/t'
txt = txt[0:-1]+'n'
print(txt)

