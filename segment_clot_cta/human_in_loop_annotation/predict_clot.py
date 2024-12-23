from segment_clot_cta.inference.inference_on_standard_dataset import predict_and_show
import os


if __name__ == '__main__':

    # predicting clot and visualization
    fold = (0, 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold[0])
    from segment_clot_cta.inference.inference_pe_v3 import load_saved_model_guided
    fn_list = os.listdir('/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/rescaled_ct-denoise')
    model = load_saved_model_guided(high_resolution=False)
    for fn in fn_list[fold[0]:: fold[1]]:
        print("processing:", fn)
        predict_and_show(high_resolution=False, file_name=fn,
                         dataset_dict='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/',
                         image_save_dict='/data_disk/temp/visualize/clot_predict_no_gt/',
                         model_loaded=model,
                         save_dict_clot_mask='/data_disk/pulmonary_embolism/segment_clot_on_CTA/PE_CTA_no_gt/'
                                             'rescaled_clot_predict')
    exit()

    # undo spatial rescale and check alignment

