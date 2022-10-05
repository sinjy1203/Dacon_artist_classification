## import
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
