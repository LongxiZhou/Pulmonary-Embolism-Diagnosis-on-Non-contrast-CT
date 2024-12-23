from pulmonary_nodules.predict_pipeline import rescaled_ct_to_semantic_seg


if __name__ == '__main__':
    dict_rescaled_ct = '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_ct/non-contrast/'
    dict_semantic = '/home/zhoul0a/Desktop/pulmonary_embolism/rescaled_masks/non-contrast/'
    rescaled_ct_to_semantic_seg(dict_rescaled_ct,
                                dict_semantic, artery_vein=False, batch_size=2, fold=(0, 3))
