from PIL import Image

from globals import *

def image_crop():
    numb = str(datetime.datetime.now())
    numb = numb.replace(":", "_")
    img = Image.open(IMG)
    res = img.crop((0, 0, 320, 240))
    res.save(f"{numb}.jpg")
    print("Cropped Foto gespeichert")

