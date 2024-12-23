import numpy as np
import sys
sys.path.append('/home/zhoul0a/Desktop/Longxi_Platform')
import Tool_Functions.Functions as Functions


def posterior_prob(rer_value, positive_new=None, negative_new=None):
    if positive_new is None or negative_new is None:
        positive_new = Functions.pickle_load_object(
            '/data_disk/pulmonary_embolism_final/pickle_objects/rer_for_PE_positives.pickle')
        negative_new = Functions.pickle_load_object(
            '/data_disk/pulmonary_embolism_final/pickle_objects/rer_for_PE_negatives.pickle')
    if rer_value > np.max(negative_new):
        return 1

    true_pos = np.sum(np.array(positive_new) > rer_value)
    false_pos = np.sum(np.array(negative_new) > rer_value)

    return true_pos / (false_pos + true_pos)

