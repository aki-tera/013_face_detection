import cv2
import matplotlib.pyplot as plt

image_path = "publicdomainq-0040727kna.jpg"
result_path = "result.jpg"
cascade_fp2="***your haarcascade_frontalface_default.xml**"
#haarcascade_frontalface_default.xml is in th "Python3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

origin_img= cv2.imread(image_path)
grayscale_img = cv2.cvtColor(origin_img,cv2.COLOR_RGB2GRAY)


cascade = cv2.CascadeClassifier(cascade_fp2)
front_face_list = cascade.detectMultiScale(grayscale_img,minSize =(100,100))

print(front_face_list)

#There is a problem that processing is doubled if there are multiple frames
for(x,y,w,h)in front_face_list:
#X-Y coordinates of the frame 
    print("[x,y] = %d,%d[w,h] = %d,%d"%(x,y,w,h))
#cut image
    cut_img = origin_img[x:x+w, y:y+h]
    cv2.imwrite(result_path, cut_img)
#drow frame
    cv2.rectangle(origin_img,(x,y),(x+w,y+h),(0,0,255),thickness = 10)
    
    plt.imshow(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB))
    plt.show()
