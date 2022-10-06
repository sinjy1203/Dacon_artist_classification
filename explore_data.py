## import
import glob
import PIL
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
shape_lst = [np.mean(np.array(PIL.Image.open(img_dir)).shape) for img_dir in img_dirs]

##
df = pd.DataFrame({'shape': shape_lst})
cnt_ser = df['shape'].value_counts(bins=10)
x = [str(int(interval.mid)) for interval in cnt_ser.index]
y = cnt_ser.values
plt.bar(x, y)
plt.show()
