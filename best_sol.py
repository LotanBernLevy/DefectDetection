from utils import *


def preprocess(inspected_image, reference_image):
    denoised_ins, denoised_ref = denoise([inspected_image], [reference_image], filter_strength=9.0)

    ins_aligner = AlignHelper(denoised_ins[0])
    ref_aligner = AlignHelper(denoised_ref[0])
    aligned_ins, h = ins_aligner.align_to(ref_aligner)

    adjusted_ref = ins_aligner.adjust_zeros_from_alignment(ref_aligner, h)
    return aligned_ins, adjusted_ref, h


def detect_defects(inspected_image, reference_image):
    aligned_ins, adjusted_ref, h = preprocess(inspected_image, reference_image)

    ssim_helper = SSIM_helper(aligned_ins, adjusted_ref)
    ssim_helper.fix_by_alignments(move_max=40, min_crop_size=50)

    defect_map = cv.cvtColor(np.zeros_like(aligned_ins), cv.COLOR_GRAY2BGR)
    defect_map = ssim_helper.draw(defect_map, color=(255, 255, 255))
    height, width = inspected_image.shape
    return cv.warpPerspective(defect_map, np.linalg.inv(h), (width, height))


if __name__ == "__main__":
    inspected_path = "data\\defective_examples\\case1_inspected_image.tif"
    reference_path = "data\\defective_examples\\case1_reference_image.tif"
    inspected_image = read_image(inspected_path)
    reference_image = read_image(reference_path)
    defect_map = detect_defects(inspected_image, reference_image)
    plot_images([inspected_image], [defect_map])
