import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc
import math

block_to_no_channels = {1: 16, 2:23, 3:32, 4:0}

def get_img(src, img_size = False):
        img = scipy.misc.imread(src, mode ="RGB")
        if not (len(img.shape) == 3 and img.shape[2] ==3):
            img = np.dstack(img, img, img)
        if img_size != False:
            img = scipy.misc.imresize(img, img_size) 
        return img   

def crop_image(img, block, channel):
        no_channels = block_to_no_channels[block]
        width, height, _ = img.shape

        if width == height:
            mini_img_width, mini_img_height = math.floor(width/no_channels), math.floor(height/no_channels)
            pad_x, pad_y = width-(no_channels*mini_img_width), height-(no_channels*mini_img_height)
            if pad_x == no_channels-1:
                pad_x, pad_y = 1,1

            # (width-no_channels+1)/no_channels, (height-no_channels+1)/no_channels
            #print(mini_img_width, mini_img_height)
        else: 
            print("Could not detect number of images. Width %s and Height %s." % (width, height))	
            mini_img_width, mini_img_height = math.floor(width/no_channels), math.floor(height/no_channels)
            pad_x, pad_y = width-(no_channels*mini_img_width), height-(no_channels*mini_img_height)
            if pad_x == no_channels-1:
                pad_x, pad_y = 1,1

        y = 0

        x = math.floor(channel/no_channels) * width + (channel % no_channels)* mini_img_width+ ((channel % no_channels)-1)*pad_x

        #x = channel*mini_img_width+channel-1
        #print(y)
        while x > width:
            y += mini_img_height+pad_y
            x -= width

        x_0 = x - mini_img_width
        x_1 = x

        y_0 = y#y/mini_img_width
        y_1 = y_0+mini_img_width

        #print(y_0, y_1)
        #print(x,y )	
        cropped_img = img[y_0:y_1, x_0:x_1]
        #cropped_img = np.array(cropped_img)
        #print(cropped_img.shape,"!!!!!!!!!!!!")
        # 	cropped_img = cropped_img[0:mini_img_width][:][0:28]
        #print(cropped_img.shape)
        #plt.imshow(cropped_img)
        #plt.axis('off')
        #plt.show()
        return cropped_img
    #plt.tight_layout(pad=0)
    #plt.savefig("../BSc thesis/arbeit/figures/%s.png" % folder, bbox_inches = 'tight', pad_inches=0)
    #plt.close()
    #plt.show()

def gallery(array, ncols=3):
    print(array.shape)
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity))
            .swapaxes(1,2)
            .reshape((height*nrows, width*ncols, intensity)))
    return result



if __name__ == "__main__":
    folders = ["Block1_14", "Block1_25", "Block1_71", "Block2_0", "Block2_47", "Block2_79", "Block3_14", "Block3_21", "Block3_597", "Block1_14_dc", "Block1_25_dc", "Block1_71_dc", "Block2_0_dc", "Block2_47_dc", "Block2_79_dc", "Block3_14_dc", "Block3_21_dc", "Block3_597_dc" ]
    for appendix in folders:
        path = ("%s" % appendix)
        array = []
        all_files = os.listdir(path)
        all_files = sorted(all_files)
        for file in all_files:
            #print(file)
            file_path = "%s/%s" % (path, file)
            f = file.split("_")
            block =int(f[0][-1])
            channel = int(f[1])
            #print(block, channel, file_path)
            img = get_img(file_path)   
            img_cropped = crop_image(img = img, block=block, channel = channel)
            #print(img_cropped.shape)

            array.append([img_cropped])

        np_array =[]
        for i in range(0,len(array),9):
            np_array.append(np.squeeze(np.array([array[i],array[i+1],array[i+2],array[i+3],array[i+4],array[i+5],array[i+6],array[i+7],array[i+8]]),1))
            #array = np.array(array)
            #array = np.squeeze(array, 1)

    
        for i,array in enumerate(np_array):

            #array = array[9:]
            result = gallery(array)
            plt.imshow(result)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig("final_images/%s%s.png" % (appendix,i), bbox_inches = 'tight', pad_inches=0)
            plt.close()
            #plt.show()

