## import
import glob
import PIL
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms


##
data_dir = Path('C:/Users/sinjy/PycharmProjects/data/dacon_artist')
train_csv = pd.read_csv(data_dir / 'train.csv')
test_csv = pd.read_csv(data_dir / 'test.csv')
artist_info = pd.read_csv(data_dir / 'artists_info.csv')

##
train_csv['artist'].value_counts().plot(kind='bar')
plt.show()

##
img_dirs = glob.glob("./data/train/*")
img = PIL.Image.open(img_dirs[0])

##
shape_lst = [np.array(PIL.Image.open(img_dir)).shape for img_dir in img_dirs]

##
lst = list(filter(lambda x: len(x) == 3, shape_lst))

##
from tqdm import tqdm
for i in tqdm(range(10, 14)):
    print(i)

##
for img_dir in img_dirs:
    if len(np.array(PIL.Image.open(img_dir)).shape) == 2:
        gray_img_dir = img_dir
        break

##
gray_img = np.array(PIL.Image.open(gray_img_dir))
plt.imshow(gray_img)
plt.show()

##
totensor = transforms.ToTensor()
scale = transforms.Scale((500, 500))

scaled_gray_img = scale(PIL.Image.open(gray_img_dir))
plt.imshow(scaled_gray_img)

##
gray_tensor = totensor(scaled_gray_img)

##
a, b, c = gray_tensor.repeat(3, 1, 1)

##
a_n = a.numpy()
b_n = b.numpy()
plt.subplot(121)
plt.imshow(a_n)
plt.subplot(122)
plt.imshow(b_n)
plt.show()

##
df = pd.DataFrame({'shape': shape_lst})
cnt_ser = df['shape'].value_counts(bins=10)
x = [str(int(interval.mid)) for interval in cnt_ser.index]
y = cnt_ser.values
plt.bar(x, y)
plt.show()
