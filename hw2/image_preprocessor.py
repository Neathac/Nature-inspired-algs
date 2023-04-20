import PIL
import os
import os.path
from PIL import Image

path = 'dataset/methamphetamine'

for file in os.listdir(path):
    f_img = path+"/"+file
    img = Image.open(f_img)
    img = img.resize((224,224))
    img.save(f_img)

