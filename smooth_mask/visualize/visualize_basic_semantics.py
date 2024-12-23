from chest_ct_database.visualize_manager.add_basic_tissue_visualize import add_visualization_basic_semantic

if __name__ == '__main__':
    add_visualization_basic_semantic(
        '/data_disk/artery_vein_project/extract_blood_region/rescaled_ct-denoise/',
        '/data_disk/artery_vein_project/extract_blood_region/semantics/',
        '/data_disk/artery_vein_project/extract_blood_region/visualization/basic_semantic_check/', fold=(0, 1))
