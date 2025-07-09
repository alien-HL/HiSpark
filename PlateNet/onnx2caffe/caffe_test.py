#coding = utf-8
import numpy as np
import sys
sys.path.append('/home/nicta100-s26/ai/sample/yolov3/hisi_yolo/caffe/python')
import caffe
import cv2

deploy_proto = '/home/nicta100-s26/ai/sample/ocr/crnn_plate_recognition/saved_model/best.prototxt'
caffe_model = '/home/nicta100-s26/ai/sample/ocr/crnn_plate_recognition/saved_model/best.caffemodel'
img = '/home/nicta100-s26/ai/sample/ocr/crnn_plate_recognition/images/test.jpg'
plate_chr="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"

# relu 激活函数
def relu(array):
    return np.maximum(array,0)

net = caffe.Net(deploy_proto, caffe_model, caffe.TEST)
im = cv2.imread(img)

# 图片预处理 变更为模型中的尺寸
im = cv2.resize(im, (168,48))
mean_value,std_value=(0.588,0.193)
im=(im / 255. - mean_value) / std_value
im = im.transpose([2, 0, 1])
im=im.reshape(1,3,48,168)

#执行上面设置的图片预处理操作，并将图片载入到blob中
net.blobs['images'].data[...] = im
out= net.forward()['output']
res=np.argmax(out[0],1)

#对输出结果进行解码
pre=0
newPreds=[]
for i in range(len(res)):
    if res[i]!=0 and res[i]!=pre:
        newPreds.append(res[i])
    pre=res[i]
plate=""
for i in newPreds:
    plate+=plate_chr[int(i)]

#输出检测结果
print(plate)
