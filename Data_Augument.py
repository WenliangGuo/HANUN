
from PIL import Image
import os 
import glob 

def get_image_paths(folder,fmt): 
    return glob.glob(os.path.join(folder, fmt)) 

def create_read_img(filename,fmt): 
    #读取图像
    im = Image.open(filename)

    out_h = im.transpose(Image.FLIP_LEFT_RIGHT)
    out_w = im.transpose(Image.FLIP_TOP_BOTTOM)
    out_90 = im.transpose(Image.ROTATE_90)
    out_180 = im.transpose(Image.ROTATE_180)
    out_270 = im.transpose(Image.ROTATE_270)
    
    out_h.save(filename[:-4]+'_h.'+fmt)
    out_w.save(filename[:-4]+'_w.'+fmt)
    out_90.save(filename[:-4]+'_90.'+fmt)
    out_180.save(filename[:-4]+'_180.'+fmt)
    out_270.save(filename[:-4]+'_270.'+fmt)
    print(filename)
    
img_path_gt = 'train_data/CUHK_train_gt/' 
imgs_gt = get_image_paths(img_path_gt,'*.png') 

for i in imgs_gt: 
    create_read_img(i,'png')

img_path_src = 'train_data/CUHK_train_source/'
imgs_src = get_image_paths(img_path_src,'*.jpg')

for i in imgs_src: 
    create_read_img(i,'jpg')


