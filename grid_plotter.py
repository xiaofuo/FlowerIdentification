import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc

def get_img(src, img_size = False):
    img = scipy.misc.imread(src, mode ="RGB")
    if not (len(img.shape) == 3 and img.shape[2] ==3):
        img = np.dstack(img, img, img)
    if img_size != False:
        img = scipy.misc.imresize(img, img_size) 
    return img   

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1,2)
              .reshape((height*nrows, width*ncols, intensity)))
    return result

def make_array(folder):
    from PIL import Image
    
    
    array = []
    for filename in os.listdir(folder):
        filename = os.path.join(folder, filename)
        img = get_img(filename, img_size=[1000, 1000])
        array.append([np.asarray(img)])

    array = np.array(array)
    array = np.squeeze(array, 1)

    print(array.shape)
    return array
   
folders = ["block2_unit4_channel0", "block1_unit3_channel14", "block1_unit3_channel25", "block1_unit3_channel71", "block2_unit4_channel47", "block2_unit4_channel79", "block3_unit6_channel14", "block3_unit6_channel21", "block3_unit6_channel597", "block4_unit3_channel9", "block4_unit3_channel12", "block4_unit3_channel35"]

for folder in folders:    
    array = make_array(folder)
    result = gallery(array)
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig("../BSc thesis/arbeit/figures/%s.png" % folder, bbox_inches = 'tight', pad_inches=0)
    plt.close()
    #plt.show()