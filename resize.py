import glob
from PIL import Image

src_folder = 'vis'

files = glob.glob(src_folder + '/*/*.*')

for file in files:

    img = Image.open(file)
    resized_img = img.resize((320,240))

    filename = file.split('/')[-1]
    resized_img.save(file) # 保存回原文件

print('Images resized!')