"""
for a scan with CTPA:
if 'clot_volume_artery' > XX and 'avr' > XX and 'artery_volume' > XX and 'vein_volume' > XX and :
   -> PE_positive
else:
   -> Uncertain:

"""
import Tool_Functions.Functions as Functions
import numpy as np
import os


def get_pe_scan_name_and_relative_importance(pickle_path='/data_disk/pulmonary_embolism_final/pickle_objects/'
                                                         'scan_name_clot_metric_dict.pickle'):
    """

    metrics calculated with CTPA

    :param pickle_path:
    :return: (scan_name, importance)
    """
    scan_name_clot_metric_dict = Functions.pickle_load_object(pickle_path)

    def classify_type(clot_metric_dict):
        if clot_metric_dict['human_annotation'] and clot_metric_dict['clot_volume'] > 0:  # only annotate PE positive
            return True
        if clot_metric_dict['clot_volume_artery'] < 200:
            return False
        if not clot_metric_dict['avr'] > 4:
            return False
        if clot_metric_dict['artery_volume'] < 250000:
            return False
        if clot_metric_dict['vein_volume'] < 150000:
            return False
        return True

    def determine_importance(clot_metric_dict):
        if clot_metric_dict['human_annotation']:
            return 2
        if clot_metric_dict['avr'] > 10:
            return 1
        return 0.75 ** (10 - clot_metric_dict['avr'])

    name_set = scan_name_clot_metric_dict.keys()

    pe_name_importance_list = []
    for scan_name in name_set:
        metric_dict = scan_name_clot_metric_dict[scan_name]
        if classify_type(metric_dict):
            pe_name_importance_list.append((scan_name, determine_importance(metric_dict)))
    return pe_name_importance_list


def statics_human_annotation(pickle_path='/data_disk/pulmonary_embolism_final/pickle_objects/'
                                         'scan_name_clot_metric_dict.pickle'):
    scan_name_clot_metric_dict = Functions.pickle_load_object(pickle_path)
    name_set = scan_name_clot_metric_dict.keys()
    avr_list = []
    clot_volume_artery_list = []
    artery_list = []
    vein_list = []
    for scan_name in name_set:
        metric_dict = scan_name_clot_metric_dict[scan_name]
        if metric_dict['human_annotation'] and metric_dict['clot_volume'] > 0:
            avr_list.append(metric_dict['avr'])
            clot_volume_artery_list.append(metric_dict['clot_volume_artery'])
            artery_list.append(metric_dict['artery_volume'])
            vein_list.append(metric_dict['vein_volume'])

    def get_statics(list_like):
        return np.min(list_like), np.percentile(list_like, 5), np.percentile(list_like, 10), np.median(list_like), \
               np.percentile(list_like, 90), np.percentile(list_like, 95)

    print("num human annotation:", len(avr_list))

    print("avr")
    avr_list.sort()
    print(avr_list)
    print(get_statics(avr_list))
    print("\nclot volume")
    clot_volume_artery_list.sort()
    print(clot_volume_artery_list)
    print("\nartery volume")
    artery_list.sort()
    print(artery_list)
    print(get_statics(artery_list))
    print("\nvein volume")
    vein_list.sort()
    print(vein_list)
    print(get_statics(vein_list))


class ScanNameTypeDict:
    def __init__(self):
        self.name_type_dict = {}
        self._form_name_scan_type()

    def _form_name_scan_type(self):
        key_list_rad = os.listdir('/data_disk/RAD-ChestCT_dataset/rescaled_ct-denoise')
        for key in key_list_rad:
            self.name_type_dict[key[:-4]] = {"type": 'non_PE', "source": "rad"}

        from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_simulate_clot.form_raw_dataset import \
            get_top_dicts as get_top_dicts_rescaled_ct_and_depth

        for dataset in ['mudanjiang', 'yidayi', 'xwzc', 'four_center_data']:
            top_dict_ct, top_dict_depth_and_branch = get_top_dicts_rescaled_ct_and_depth(dataset, denoise=False)
            key_list = os.listdir(top_dict_ct)
            for key in key_list:
                self.name_type_dict[key[:-4]] = {"type": 'non_PE', "source": dataset}

        for scan_name, importance in get_pe_scan_name_and_relative_importance():
            self.name_type_dict[scan_name] = {"type": 'PE', "source": 'paired_dataset', 'importance': importance}

        from pe_dataset_management.basic_functions import get_all_scan_name
        for scan_name in get_all_scan_name():
            if scan_name not in self.name_type_dict.keys():
                self.name_type_dict[scan_name] = {"type": "unknown", "source": 'paired_dataset'}

    def get_name_type(self, scan_name):
        if len(scan_name) > 4:
            if scan_name[-4:] == '.npz' or scan_name[-4:] == '.npy':
                scan_name = scan_name[:-4]
        if len(scan_name) > 7:
            if scan_name[-7:] == '.pickle':
                scan_name = scan_name[:-7]
        if scan_name not in self.name_type_dict.keys():
            return None
        return self.name_type_dict[scan_name]

    def name_field(self):
        return self.name_type_dict.keys()


if __name__ == '__main__':

    dict_class = ScanNameTypeDict()
    print(dict_class.get_name_type('Z167'))
    exit()
    from pe_dataset_management.basic_functions import get_all_scan_name
    print(get_all_scan_name())
    exit()
    print(get_pe_scan_name_and_relative_importance())
    exit()
    print(len(get_pe_scan_name_and_relative_importance()))
    statics_human_annotation()
    exit()
    print(get_pe_scan_name_and_relative_importance())
    exit()
