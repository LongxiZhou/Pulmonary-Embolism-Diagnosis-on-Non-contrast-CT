import Tool_Functions.Functions as Functions
import os

"""
pickle save data structure:
{scan_name: {'label_whether_pe': bool, 'num_label_pe_slice': int, 'direction_probability': float, 'av_metrics': dict}}
"""


pickle_save_path = '/data_disk/RSNA-PE_dataset/pickle_objects/av_metrics_pe_label_v1.pickle'


def extract_pe_label_information():
    csv_path = '/data_disk/RSNA-PE_dataset/rsna-str-pulmonary-embolism-detection/train.csv'

    pickle_object = {}

    with open(csv_path, 'r') as f:
        total_pe_image_in_scan = 0
        previous_scan_name = '6897fa9de148'  # the first scan name in csv
        pickle_object[previous_scan_name] = {}
        for index, line in enumerate(f):
            if index == 0:
                continue

            values = line.split(',')
            # [StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID, pe_present_on_image, negative_exam_for_pe, ...]

            scan_name = values[0]  # str
            if scan_name == previous_scan_name:
                total_pe_image_in_scan += int(values[3])
            else:
                # move into the next scan
                # conclude the previous
                pickle_object[previous_scan_name]['num_label_pe_slice'] = total_pe_image_in_scan
                if total_pe_image_in_scan == 0:
                    pickle_object[previous_scan_name]['label_whether_pe'] = False
                else:
                    pickle_object[previous_scan_name]['label_whether_pe'] = True
                # prepare for the current scan
                total_pe_image_in_scan = 0
                previous_scan_name = scan_name
                pickle_object[scan_name] = {}

        # the last scan
        if len(previous_scan_name) > 0:
            pickle_object[previous_scan_name]['num_label_pe_slice'] = total_pe_image_in_scan
            if total_pe_image_in_scan == 0:
                pickle_object[previous_scan_name]['label_whether_pe'] = False
            else:
                pickle_object[previous_scan_name]['label_whether_pe'] = True

    Functions.pickle_save_object(pickle_save_path, pickle_object)


def merge_direction_probability_and_av_metrics(
        top_dict_av_metrics='/data_disk/RSNA-PE_dataset/pickle_objects/av_metrics_low_reso',
        top_dict_direction_probability='/data_disk/RSNA-PE_dataset/pickle_objects/direction_probability'):

    current_object = Functions.pickle_load_object(pickle_save_path)

    scan_name_list = list(current_object.keys())

    pickle_name_list_in_direction = set(os.listdir(top_dict_direction_probability))
    pickle_name_list_in_av_metrics = set(os.listdir(top_dict_av_metrics))

    valid_name_set = pickle_name_list_in_av_metrics & pickle_name_list_in_direction

    final_object = {}

    for scan_name in scan_name_list:
        pickle_name = scan_name + '.pickle'
        if pickle_name in valid_name_set:
            direction_path = os.path.join(top_dict_direction_probability, pickle_name)
            sorted_direction_probability_list = Functions.pickle_load_object(direction_path)
            if not sorted_direction_probability_list[0]['class_id'] == 0:
                continue

            metrics_path = os.path.join(top_dict_av_metrics, pickle_name)
            av_metrics_dict = Functions.pickle_load_object(metrics_path)

            final_object[scan_name] = current_object[scan_name]
            final_object[scan_name]['direction_probability'] = \
                sorted_direction_probability_list[0]['predicted_probability']
            final_object[scan_name]['av_metrics'] = av_metrics_dict

    Functions.pickle_save_object('/data_disk/RSNA-PE_dataset/pickle_objects/av_metrics_pe_label_v1_final.pickle', final_object)


if __name__ == '__main__':
    extract_pe_label_information()
    merge_direction_probability_and_av_metrics()
