import Tool_Functions.file_operations as file_operations
import os


top_dict_source = '/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_24-02-01/A436-A534'
top_dict_save = '/data_disk/CTA-CT_paired-dataset/transfer/paired_new_data_24-02-01/split_FBP_IR'
fn_list = os.listdir(top_dict_source)

for fn in fn_list:
    save_dir_cta_fbp = os.path.join(top_dict_save, fn + '_FBP', 'CTA')
    save_dir_cta_ir = os.path.join(top_dict_save, fn + '_IR', 'CTA')

    save_dir_non_fbp = os.path.join(top_dict_save, fn + '_FBP', 'non-contrast')
    save_dir_non_ir = os.path.join(top_dict_save, fn + '_IR', 'non-contrast')

    source_dict_cta_fbp = os.path.join(top_dict_source, fn, 'CTA1')
    source_dict_cta_ir = source_dict_cta_fbp

    source_dict_non_fbp = os.path.join(top_dict_source, fn, 'non-contrast1')
    source_dict_non_ir = os.path.join(top_dict_source, fn, 'non-contrast2')

    file_operations.copy_file_or_dir(source_dict_cta_fbp, save_dir_cta_fbp)
    file_operations.move_file_or_dir(source_dict_cta_ir, save_dir_cta_ir)
    file_operations.move_file_or_dir(source_dict_non_fbp, save_dir_non_fbp)
    file_operations.move_file_or_dir(source_dict_non_ir, save_dir_non_ir)

    file_operations.remove_path_or_directory(os.path.join(top_dict_source, fn))

