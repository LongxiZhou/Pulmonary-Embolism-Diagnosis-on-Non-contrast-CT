import os

top_dict_paired_dcm_dataset = '/data_disk/CTA-CT_paired-dataset/paired_dcm_files'

fn_list_pe_high_quality = os.listdir(os.path.join(top_dict_paired_dcm_dataset, 'PE_High_Quality'))
fn_list_pe_low_quality_ct_after_cta = os.listdir(
    os.path.join(top_dict_paired_dcm_dataset, 'PE_Low_Quality/CT-after-CTA'))
fn_list_pe_good_interval_thick_slice = os.listdir(
    os.path.join(top_dict_paired_dcm_dataset, 'PE_Low_Quality/good_CTA-CT_interval_but_bad_dcm'))
fn_list_pe_long_cta_ct_interval = os.listdir(
    os.path.join(top_dict_paired_dcm_dataset, 'PE_Low_Quality/long_CTA-CT_interval'))
fn_list_pe_seem_not_pair = os.listdir(os.path.join(top_dict_paired_dcm_dataset, 'may_not_pair/PE'))
fn_list_pe_strange_data = os.listdir(os.path.join(top_dict_paired_dcm_dataset, 'strange_data/PE'))

fn_list_normal_high_quality = os.listdir(os.path.join(top_dict_paired_dcm_dataset, 'Normal_High_Quality'))
fn_list_normal_low_quality_ct_after_cta = os.listdir(
    os.path.join(top_dict_paired_dcm_dataset, 'Normal_Low_Quality/CT-after-CTA'))
fn_list_normal_good_interval_thick_slice = os.listdir(
    os.path.join(top_dict_paired_dcm_dataset, 'Normal_Low_Quality/good_CTA-CT_interval_but_bad_dcm'))
fn_list_normal_long_cta_ct_interval = os.listdir(
    os.path.join(top_dict_paired_dcm_dataset, 'Normal_Low_Quality/long_CTA-CT_interval'))
fn_list_normal_seem_not_pair = os.listdir(os.path.join(top_dict_paired_dcm_dataset, 'may_not_pair/Normal'))
fn_list_normal_strange_data = os.listdir(os.path.join(top_dict_paired_dcm_dataset, 'strange_data/Normal'))


def export_excel_pe(save_path='/data_disk/CTA-CT_paired-dataset/reports/PE_report.csv'):
    head_line = ["PE_high_quality", "PE_ct_after_cta", "PE_long_cta_ct_interval", "PE_good_interval_thick_slice",
                 "PE_seem_not_pair", "PE_strange_data"]

    max_instance = max(len(fn_list_pe_high_quality), len(fn_list_pe_low_quality_ct_after_cta),
                       len(fn_list_pe_good_interval_thick_slice), len(fn_list_pe_long_cta_ct_interval),
                       len(fn_list_pe_seem_not_pair), len(fn_list_pe_strange_data))

    with open(save_path, 'w') as f:
        f.write(csv_merge_line(head_line))
        new_line = []
        for index in range(max_instance):
            new_line.append(get_item_from_fn_list(fn_list_pe_high_quality, index))
            new_line.append(get_item_from_fn_list(fn_list_pe_low_quality_ct_after_cta, index))
            new_line.append(get_item_from_fn_list(fn_list_pe_good_interval_thick_slice, index))
            new_line.append(get_item_from_fn_list(fn_list_pe_long_cta_ct_interval, index))
            new_line.append(get_item_from_fn_list(fn_list_pe_seem_not_pair, index))
            new_line.append(get_item_from_fn_list(fn_list_pe_strange_data, index))
            f.write(csv_merge_line(new_line))
            new_line = []

    f.close()


def export_excel_normal(save_path='/data_disk/CTA-CT_paired-dataset/reports/Normal_report.csv'):
    head_line = ["Normal_high_quality", "Normal_ct_after_cta", "Normal_long_cta_ct_interval",
                 "Normal_good_interval_thick_slice", "Normal_seem_not_pair", "Normal_strange_data"]

    max_instance = max(len(fn_list_normal_high_quality), len(fn_list_normal_low_quality_ct_after_cta),
                       len(fn_list_normal_good_interval_thick_slice), len(fn_list_normal_long_cta_ct_interval),
                       len(fn_list_normal_seem_not_pair), len(fn_list_normal_strange_data))

    with open(save_path, 'w') as f:
        f.write(csv_merge_line(head_line))
        new_line = []
        for index in range(max_instance):
            new_line.append(get_item_from_fn_list(fn_list_normal_high_quality, index))
            new_line.append(get_item_from_fn_list(fn_list_normal_low_quality_ct_after_cta, index))
            new_line.append(get_item_from_fn_list(fn_list_normal_good_interval_thick_slice, index))
            new_line.append(get_item_from_fn_list(fn_list_normal_long_cta_ct_interval, index))
            new_line.append(get_item_from_fn_list(fn_list_normal_seem_not_pair, index))
            new_line.append(get_item_from_fn_list(fn_list_normal_strange_data, index))
            f.write(csv_merge_line(new_line))
            new_line = []

    f.close()


def csv_merge_line(value_list):
    return_str = ''
    for value in value_list:
        return_str = return_str + value + ','
    return_str = return_str[:-1] + '\n'
    return return_str


def get_item_from_fn_list(fn_list, index):
    if index >= len(fn_list):
        return ''
    return fn_list[index]


if __name__ == '__main__':
    export_excel_pe()
    export_excel_normal()
