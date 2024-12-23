import Tool_Functions.Functions as Functions
import numpy as np
i_5mm, _ = Functions.load_dicom('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000029/5mm/IMG-0006-00030.dcm')
start_i = 144
i_0, _ = Functions.load_dicom('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000029/1mm/IMG-0005-00' + str(start_i) + '.dcm')
i_1, _ = Functions.load_dicom('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000029/1mm/IMG-0005-00' + str(start_i + 1) + '.dcm')
i_2, _ = Functions.load_dicom('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000029/1mm/IMG-0005-00' + str(start_i + 2) + '.dcm')
i_3, _ = Functions.load_dicom('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000029/1mm/IMG-0005-00' + str(start_i + 3) + '.dcm')
i_4, _ = Functions.load_dicom('/home/zhoul0a/Desktop/其它肺炎/6正常肺-233例/xwzc000029/1mm/IMG-0005-00' + str(start_i + 4) + '.dcm')

Functions.image_show(np.clip(i_5mm, -1000, 200), gray=True)
i_average = (i_0 + i_1 + i_2 + i_3 + i_4) / 5
Functions.image_show(np.clip(i_average, -1000, 200), gray=True)
Functions.image_show(i_average - i_5mm)

Functions.image_save( np.clip(i_5mm, -1000, 200), '/home/zhoul0a/Desktop/Lung_Altas/picture_reproduce/5mm_ground_truth.png', dpi=600, gray=True)
Functions.image_save(np.clip(i_average, -1000, 200), '/home/zhoul0a/Desktop/Lung_Altas/picture_reproduce/5mm_stacked_by_1mm.png', dpi=600, gray=True)
Functions.image_save(i_average - i_5mm, '/home/zhoul0a/Desktop/Lung_Altas/picture_reproduce/difference_5mm_gt&stacked.png', dpi=600)