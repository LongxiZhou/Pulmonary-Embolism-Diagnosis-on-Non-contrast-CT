import imageio
import os
top_dict = '/data_disk/pulmonary_embolism_final/pictures/segmentation_visualization'
images = []
fn_list = os.listdir(top_dict)
fn_list.sort()
for filename in fn_list:
    images.append(imageio.imread(os.path.join(top_dict, filename)))
imageio.mimsave('/data_disk/pulmonary_embolism_final/pictures/movie_seg.gif', images, duration=0.25)
