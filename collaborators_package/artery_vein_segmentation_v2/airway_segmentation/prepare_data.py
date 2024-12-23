import numpy as np
import os
from analysis.get_surface_rim_adjacent_mean import get_surface

class SplitComb():
    def __init__(self):
        """
        :param side_len: list of input shape, default=[80,192,304] \
        :param margin: sliding stride, default=[60,60,60]
        """
        self.side_len = [128, 128, 64]
        self.margin = [80, 192, 304]

    def split_id(self, data):
        """
        :param data: target data to be splitted into sub-volumes, shape = (D, H, W) \
        :return: output list of coordinates for the cropped sub-volumes, start-to-end
        """
        side_len = self.side_len
        margin = self.margin

        splits = []
        z, h, w = data.shape

        nz = int(np.ceil(float(z - self.margin[0]) / side_len[0]))
        nh = int(np.ceil(float(h - self.margin[1]) / side_len[1]))
        nw = int(np.ceil(float(w - self.margin[2]) / side_len[2]))

        # assert (nz * side_len[0] + self.margin[0] - z >= 0)
        # assert (nh * side_len[1] + self.margin[1] - h >= 0)
        # assert (nw * side_len[2] + self.margin[2] - w >= 0)

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [[0, nz * side_len[0] + self.margin[0] - z],
               [0, nh * side_len[1] + self.margin[1] - h],
               [0, nw * side_len[2] + self.margin[2] - w]]
        orgshape = [z, h, w]

        idx = 0
        for iz in range(nz + 1):
            for ih in range(nh + 1):
                for iw in range(nw + 1):
                    sz = iz * side_len[0]  # start
                    ez = iz * side_len[0] + self.margin[0]  # end
                    sh = ih * side_len[1]
                    eh = ih * side_len[1] + self.margin[1]
                    sw = iw * side_len[2]
                    ew = iw * side_len[2] + self.margin[2]
                    if ez > z:
                        sz = z - self.margin[0]
                        ez = z
                    if eh > h:
                        sh = h - self.margin[1]
                        eh = h
                    if ew > w:
                        sw = w - self.margin[2]
                        ew = w
                    idcs = [[sz, ez], [sh, eh], [sw, ew], idx]
                    splits.append(idcs)
                    idx += 1
        splits = np.array(splits)
        # split size
        return splits, nzhw, orgshape


raw_path = "/data/CT_Data_Collections/rescaled_ct/non-contrast/rescaled_ct"
airway_path = "/data/CT_Data_Collections/rescaled_ct/non-contrast/ground_truth/airway_gt"
save_path = "/data/Train_and_Test/segmentation/airway"

split_comber = SplitComb()

for filename in np.sort(os.listdir(airway_path)):
    print(filename)
    raw = np.load(os.path.join(raw_path, filename))["array"]
    airway = np.load(os.path.join(airway_path, filename))["array"]
    raw[raw == 0] = -0.25
    airway = np.array(airway > 0.5, "float32")
    airway_1 = get_surface(airway, strict=False)
    airway_2 = get_surface(airway - airway_1, strict=False)
    airway = airway_2 + airway_1

    raw = np.clip((raw * 1600 + 400) / 1400, 0, 1)

    # specify the paths
    cubelist = []
    caseNumber = 0

    splits, nzhw, orgshape = split_comber.split_id(raw)
    # print ("Name: %s, # of splits: %d"%(data_name, len(splits)))
    for j in range(len(splits)):
        cursplit = splits[j]
        # print(cursplit)
        ####################################################################
        cur_cube = raw[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                  cursplit[2][0]:cursplit[2][1]]
        gt_cube = airway[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                  cursplit[2][0]:cursplit[2][1]]
        ####################################################################

        start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
        print(start)
        normstart = ((np.array(start).astype('float') / np.array(orgshape).astype('float')) - 0.5) * 2.0
        crop_size = [cur_cube.shape[0], cur_cube.shape[1], cur_cube.shape[2]]
        stride = 1.0
        normsize = (np.array(crop_size).astype('float') / np.array(orgshape).astype('float')) * 2.0
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float')
        format_cube = np.zeros([5, 80, 192, 304])
        ####################################################################
        format_cube[0] = cur_cube
        format_cube[1] = gt_cube

        format_cube[2:] = coord

        save_name = str(j).zfill(3) + filename
        np.savez_compressed(os.path.join(save_path, save_name), format_cube)
