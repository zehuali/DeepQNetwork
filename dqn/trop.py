from PIL import Image

n = 15157
# n = 1

for x in range(n):
    img = Image.open('train_data/1/snaps/Super Mario Bros (E)-'+str(x)+'.png')
    area = (0, 8, 256, 232)
    new_img = img.crop(area)
    new_img.save('train_data/1/snaps/mario-'+str(x)+'.png')