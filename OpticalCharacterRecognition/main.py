import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

from rembg import remove
from PIL import Image

img_path = 'image3.jpg'
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(img_path)
# print(result[0])

img = cv2.imread(img_path)
Img = Image.fromarray(img)
W,H = Img.size[0], Img.size[1]
col = Img.getcolors(W*H)
col = sorted(col)
bgCol = col[len(col)//2][1]
print()
print(f"<div style='font-family: Verdana; display:inline-block; background-color:rgb{bgCol}'>")


for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 5)
    # img = cv2.putText(img, text, top_left, font, .5, (255,255,255), cv2.LINE_AA)
    cropped = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # print(top_left, bottom_right)
    # print(cropped.shape)

    croppedImg = Image.fromarray(cropped)
    w,h = croppedImg.size[0], croppedImg.size[1]
    colors = croppedImg.getcolors(w*h)
    colors = sorted(colors)
    textColor = colors[-2][1]
    print(f"<span style='font-size:{h/4}px; color:rgb{textColor}'>{text}</span><br>")

    # output = remove(Image.fromarray(cropped))
    # bgRemoved = np.array(output)
    # arr = bgRemoved.reshape(-1,4).tolist()
    # filtered = []
    # for x in arr:
    #     if not(x == [0,0,0,0]):
    #         filtered.append(x[:3])
    # m = max(filtered, key=filtered.count)
    # print(m)
    # print(arr)
    # exit()

    # print(cropped.reshape(-1,3))
    # arr = cropped.reshape(-1,3).tolist()
    # print(arr)
    # print(max(arr, key=arr.count))

# plt.figure(figsize=(10,10))
# plt.imshow(img)
# plt.show()

print("</div>")
print()