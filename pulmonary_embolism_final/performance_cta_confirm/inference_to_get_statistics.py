import Tool_Functions.Functions as Functions
import pulmonary_embolism_final.inference.predict_clot_from_sample_sequence as predict_clot
from pulmonary_embolism_final.prepare_training_dataset.prepare_dataset_with_gt_cta_confirm.scan_name_diagnosis_type \
    import ScanNameTypeDict
import os


def load_model(test_id, augment, model_type):
    if model_type == 'new_model':
        model_dir = '/data_disk/pulmonary_embolism_final/check_point_dir/use_cta_confirm'

    elif model_type == 'old_model':
        model_dir = '/data_disk/pulmonary_embolism_final/check_point_dir/use_given_pe_classify'
    else:
        assert model_type == 'simulate_only'
        if augment:
            model_path = '/data_disk/pulmonary_embolism_final/check_point_dir/warm_up_simulation_only_' \
                         'high_reso_augment/vi_0.015_dice_0.792_precision_phase_model_guided.pth'
        else:
            model_path = '/data_disk/pulmonary_embolism_final/check_point_dir/warm_up_simulation_only_' \
                         'high_reso_not_augment/vi_0.014_dice_0.720_precision_phase_model_guided.pth'
        return predict_clot.predict.load_saved_model_guided(model_path=model_path)

    if augment:
        model_dir = os.path.join(model_dir, 'high_resolution_with_augment')
    else:
        model_dir = os.path.join(model_dir, 'high_resolution')
    model_path = os.path.join(model_dir, 'with_annotation_test_id_' + str(test_id), 'best_model_guided.pth')

    return predict_clot.predict.load_saved_model_guided(model_path=model_path)


def process_test_id(test_id, trim_length=4000, augment=False, model_type='new_model', fold=(0, 1)):
    sample_dir_pe = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation_cta_confirm/' \
                    'pe_vessel_high_recall/high_resolution/pe_not_trim_not_denoise'
    sample_dir_non_pe = '/data_disk/pulmonary_embolism_final/samples_for_performance_evaluation/non_pe' \
                        '/high_resolution/not_pe_not_trim_not_denoise'

    save_statistic_dir = '/data_disk/pulmonary_embolism_final/statistic_cta_confirm'
    save_statistic_dir = os.path.join(save_statistic_dir, model_type)
    if augment:
        save_statistic_dir = os.path.join(save_statistic_dir, 'augment_trim_' + str(trim_length))
    else:
        save_statistic_dir = os.path.join(save_statistic_dir, 'not_augment_trim_' + str(trim_length))

    model = load_model(test_id, augment, model_type)

    scan_name_type_dict = ScanNameTypeDict()

    def process_sample_name(scan_name_pickle):
        sample_info = scan_name_type_dict.get_name_type(scan_name_pickle)
        if sample_info is None:
            print("sample not exist")
            return None

        save_statistic_path = os.path.join(
            save_statistic_dir, sample_info['type'] + '_' + sample_info["source"], scan_name_pickle)
        if os.path.exists(save_statistic_path):
            print("processed")
            return None
        print("sample information:", sample_info)
        if sample_info['type'] == 'non_PE':
            path_sample = os.path.join(sample_dir_non_pe, scan_name_pickle)
        else:  # unknown of PE
            path_sample = os.path.join(sample_dir_pe, scan_name_pickle)

        sample = Functions.pickle_load_object(path_sample)

        statistic = predict_clot.predict_on_evaluate_sample(
            sample, model, trim_length=trim_length, visualize=False, show_statistic=False)

        print(scan_name_pickle, statistic)

        Functions.pickle_save_object(save_statistic_path, statistic)
        return None

    def get_sample_name_pickle_test_id():
        scan_name_list_pe = os.listdir(sample_dir_pe)
        scan_name_list_non_pe = os.listdir(sample_dir_non_pe)
        scan_name_all = scan_name_list_non_pe + scan_name_list_pe
        scan_name_all = Functions.split_list_by_ord_sum(scan_name_all, fold=fold)
        scan_name_test_id = []
        for scan_name_pickle in scan_name_all:
            if Functions.get_ord_sum(scan_name_pickle) % 5 == test_id and \
                    scan_name_pickle[:-7] in scan_name_type_dict.name_field():  # name.pickle
                scan_name_test_id.append(scan_name_pickle)
        return scan_name_test_id

    name_list = get_sample_name_pickle_test_id()
    processed_count = 0
    for name in name_list:
        print("processing:", name, processed_count, '/', len(name_list))
        process_sample_name(name)
        processed_count += 1


def process_all(test_id, fold=(0, 1)):
    for augment in [True, False]:
        for model_type in ['new_model', 'old_model', 'simulate_only']:
            for trim_length in [4000, 3000]:
                process_test_id(test_id, trim_length=trim_length, augment=augment, model_type=model_type, fold=fold)


if __name__ == '__main__':
    Functions.set_visible_device('1')
    process_all(test_id=0)
