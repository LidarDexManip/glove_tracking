"""Build quality-filtered training crops from synchronized HF SAM 3.1 delivery."""
from gaussian_glove.geometry_inputs import camera_matrix, choose_indices, subject_betas, vertices_for_session, make_contact_sheet
from gaussian_glove.prepare_sam31_dataset import main
if __name__ == '__main__': main()
