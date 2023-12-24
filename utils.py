from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
from typing import Union, Tuple, List, Generator
import re
import skimage
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import warnings

INSPECTED_IMG_NAME = "inspected"
REFERENCE_IMG_NAME = "reference"





# ==============================#
# read and paths manipulations #
# ==============================#
def read_image(path: str) -> np.array:
    return np.array(Image.open(path))


def filter_inspected(images_paths: List[str]) -> List[str]:
    return [path for path in images_paths if INSPECTED_IMG_NAME in os.path.basename(path)]


def read_images_pairs(images_dir: str) -> Tuple[List[np.array], List[np.array], List[str]]:
    inspected_path = filter_inspected([os.path.join(images_dir, path) for path in os.listdir(images_dir)])
    inspected_images = [read_image(path) for path in inspected_path]
    reference_images = [read_image(path.replace(INSPECTED_IMG_NAME, REFERENCE_IMG_NAME)) for path in inspected_path]
    names = [os.path.basename(path).split("_")[0] for path in inspected_path]
    return inspected_images, reference_images, names


def read_defects(defects_file: str) -> dict:
    """Reads defect from a file
    """
    defects_dicts = dict()
    last_group = None
    with open(defects_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("defect"):
                m = re.search("x=([\d]*)", line)
                x = int(m.group(1))
                m = re.search("y=([\d]*)", line)
                y = int(m.group(1))
                if last_group is None:
                    raise ValueError("Bad defects file content")
                defects_dicts[last_group].append((x, y))
            elif line != "":
                line = line.replace(" ", "").replace(":", "")
                defects_dicts[line] = []
                last_group = line
    return defects_dicts


# =============#
# plots utils #
# =============#
def plot_images(*args, row_names: Tuple = None, save_path: str = False, size: Tuple = (10, 10), title: str = None):
    """Plot a rows of images

    Args:
        *args: Lists of images, each list represent a different column
        row_names (Tuple, optional): A tuple of lists. Each list represents the names of the images in a row. If None doesn't display images names. Defaults to None.
        save_path (str, optional): "A path to save the plot. If set to False, the output image will be displayed instead. Defaults to False.
        size (Tuple, optional): The output figure size. Defaults to (10, 10).
        title (str, optional): The title of the output image. Defaults to None.
    """
    cols = len(args)
    rows = len(args[0])

    figure = plt.figure(figsize=size)
    if title is not None:
        plt.suptitle(title, color="red")
        plt.axis("off")
    for i in range(rows):
        for j, l in enumerate(args):
            figure.add_subplot(rows, cols, cols * i + j + 1)
            if row_names is not None:
                plt.title(f"{row_names[j][i]}")
            plt.imshow(args[j][i].squeeze(), cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def display_data_pairs(images_dir, save_path=False):
    inspected_images, reference_images, names = read_images_pairs(images_dir)

    plot_images(inspected_images, reference_images, row_names=(
        [f"{name}_{INSPECTED_IMG_NAME}" for name in names], [f"{name}_{REFERENCE_IMG_NAME}" for name in names]))


def display_pairs_subtractions(inspected_images, reference_images, names, save_path: bool = False):
    """Displays the subtraction between all the reference and the inspected pairs in the given images dir.

    Args:
        images_dir (str): an images directory path
        save_path (bool): If true save the output plot instead of displaying it. Defaults to False.
    """

    plot_images([(ref - inf) for ref, inf in zip(inspected_images, reference_images)], row_names=(names))


def add_defects(images, names, defect_dict, homograpies=None):
    if homograpies is None:
        homograpies = [None] * len(images)
    color = (255, 0, 0)
    with_defects = []
    for name, img, h in zip(names, images, homograpies):
        name = name
        if name in defect_dict:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            defects = defect_dict[name]
            if h is not None:
                defects = AlignHelper.apply_homography(np.array(defects), h, img.shape[:-1]).astype(np.int)

            for coord in defects:
                img = cv.circle(img, coord, 10, color=color)
        with_defects.append(img)
    return with_defects


# ===================#
# Images Alignments #
# ===================#
FEATURES_PARAMS = dict(maxCorners=100,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7)

LK_PARAMS = dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURES_COLOR = (238, 75, 43)
MAX_FEATURES = 500  # Max descriptors allowed to generate for an image
KEEP_PERCENT = 0.15  # The percent of the matched descriptors allowed to use for alignments


class AlignHelper:
    """
    Utility object for images alignment and descriptors analyzing

    """

    MIN_DESC_NUM = 27  # Minimum descriptors allowed for alignment.

    def __init__(self, gray_img, max_features=MAX_FEATURES):
        orb = cv.ORB_create(max_features)
        self.img = gray_img
        self.kpts, self.desc = orb.detectAndCompute(gray_img, None)

    def display_kpts(self, img: np.array = None) -> np.array:
        """Use the SIFT keypoints generated for the image that provided in the constructor, and display them on the given image. 
        If the given image is None it'll depict this keypoints on the original image.

        Args:
            img (np.array, optional): A gray image to display the keypoints on. Defaults to None.

        Returns:
            np.array: BGR image with the keypoints 
        """
        if img is None:
            img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        for kpt in self.kpts:
            x, y = kpt.pt
            cv.circle(img, (int(x), int(y)), 2, color=(255, 0, 0))
        return img

    def split2clusters(self, img: np.array = None) -> np.array:
        """Generate a new image with the keypoints generated in this object constructor split into clusters by an agglomerative clustering algorithm.

        Args:
            img (np.array, optional): Image to display the keypoints on. Defaults to None.

        Returns:
            np.array: BGR image with the keypoints, that are colored according to the clusters.
        """
        if img is None:
            img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        ins_kpts = np.array([np.array([kpt.pt[0], kpt.pt[1]]) for kpt in self.kpts])
        cluster_obj = AgglomerativeClustering().fit(ins_kpts)
        rand_colors = [tuple(int(np.random.uniform(0, 255)) for _ in range(3)) for i in
                       range(np.max(cluster_obj.labels_) + 1)]
        for pt, label in zip(ins_kpts, cluster_obj.labels_):
            x, y = pt
            cv.circle(img, (int(x), int(y)), 2, color=rand_colors[label])
        return img

    @staticmethod
    def has_min_desc_num(gray_img: np.array) -> bool:
        """Check if the given image allowed to generate enough descriptors

        Args:
            gray_img (np.array): A gray image to consider.

        Returns:
            bool: If the given image has enough descriptors or not
        """
        orb = cv.ORB_create(AlignHelper.MAX_FEATURES)
        kpts, desc = orb.detectAndCompute(gray_img, None)
        return desc is not None and len(desc) >= AlignHelper.MIN_DESC_NUM

    @staticmethod
    def get_homography(aligner1, aligner2, keep_percent: float = KEEP_PERCENT) -> np.array:
        """Generate homography matrix for perspective transform for the images of the given two alignment objects.

        Args:
            aligner1 (AlignHelper): alignment object represent an image to align.
            aligner2 (AlignHelper): alignment object represent an image to align with.
            keep_percent (float, optional): The percent of matching that are allowed to use for alignment. Defaults to KEEP_PERCENT.

        Returns:
            np.array: The homography matrix
        """
        if aligner1.desc is None or len(aligner1.desc) < AlignHelper.MIN_DESC_NUM or aligner2.desc is None or len(
                aligner2.desc) < AlignHelper.MIN_DESC_NUM:
            return None
        matches = AlignHelper.get_matching_list(aligner1, aligner2, keep_percent=keep_percent)
        if len(matches) < AlignHelper.MIN_DESC_NUM:
            return None

        points_aligner1 = np.zeros((len(matches), 2), dtype=np.float32)
        points_aligner2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points_aligner1[i, :] = aligner1.kpts[match.queryIdx].pt
            points_aligner2[i, :] = aligner2.kpts[match.trainIdx].pt

        h, mask = cv.findHomography(points_aligner1, points_aligner2, cv.RANSAC)
        return h

    @staticmethod
    def get_matching_list(aligner1, aligner2, keep_percent: float = None) -> np.array:
        """Generate a matching list for the descriptors of the given two alignment objects.

        Args:
            aligner1 (AlignHelper): alignment object represent an image to align.
            aligner2 (AlignHelper): alignment object represent an image to align with.
            keep_percent (float, optional): The percent of matching that are allowed to use for alignment, If None return all matches. Defaults to None.

        Returns:
            np.array: An array of OpenCV matches objects
        """
        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(aligner1.desc, aligner2.desc, None)
        if keep_percent is not None:
            matches = sorted(matches, key=lambda x: x.distance)
            keep = max(int(len(matches) * keep_percent), AlignHelper.MIN_DESC_NUM)
            matches = matches[:keep]
        return matches

    @staticmethod
    def apply_homography(points: np.array, h: np.array, img_size: tuple):
        """Applay homography on the given points list

        Args:
            points (np.array): an array with shape n x 2
            h (np.array): homography to use
            img_size (tuple): a tuple of width x height represent the shape of the image to align with.

        Returns:
            _type_: _description_
        """

        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

        rows, cols = img_size
        new_values = np.dot(h, points.T)

        new_values /= new_values[2]
        new_values = new_values[:2].T
        new_values[new_values < 0] = 0
        new_values[new_values[:, 0] > cols, 0] = cols
        new_values[new_values[:, 1] > rows, 1] = rows
        return new_values

    def draw_matches_to(self, other, keep_percent: float = KEEP_PERCENT) -> np.array:
        """Draw the matches for alignment with the given AlignHelper object on the image that this object represents.

        Args:
            other (AlignHelper): alignment object represent an image to align with.
            keep_percent (float, optional): The percent of matching that are allowed to use for alignment, If None return all matches. Defaults to KEEP_PERCENT.

        Returns:
            float: The image of this object with the matches for alignment with the given AlignHelper objects.
        """
        matches = AlignHelper.get_matching_list(self, other, keep_percent=keep_percent)
        return cv.drawMatches(self.img, self.kpts, other.img, other.kpts, matches, None)

    def filter_best_matches_kpts(self, other, keep_percent=KEEP_PERCENT, filter_closed_also=False, queryIdx=True,
                                 best=False):
        matches = AlignHelper.get_matching_list(self, other, keep_percent=keep_percent)
        orig_kpts = self.kpts if queryIdx else other.kpts

        matches_idx = [m.queryIdx if queryIdx else m.trainIdx for m in matches]
        mask = np.ones(len(orig_kpts), dtype=bool)
        mask[matches_idx] = False
        relevant_indices = np.where(~mask if best else mask)[0]

        filtered_pt = np.zeros((len(orig_kpts) - len(matches), 2), dtype=np.float32)
        for j, i in enumerate(relevant_indices):
            filtered_pt[j, :] = orig_kpts[i].pt

        return filtered_pt

    def adjust_zeros_from_alignment(self, other, homography: np.array, img: np.array = None):
        """Copy the empty pixels generated after align the image of this current AlignHelper with the image represented by the given other AlignHelper, to another image. (To makes them similar)

        Args:
            other (AlignHelper): The AlignHelper represents the image to align with.
            homography (np.array): homography that was used for alignment
            img (np.array, optional): The image to copy the empty pixels, If None it'll use the image of the other AlignHelper. Defaults to None.

        Returns:
            np.array: The image with the empty pixels
        """
        if img is None:
            img = other.img
        if homography is None:
            return other.img
        h, w = self.img.shape
        orig_corners = np.array(np.meshgrid([0, w], [0, h])).T.reshape(-1, 2)
        new_corners = AlignHelper.apply_homography(orig_corners, homography, other.img.shape).astype(np.int)
        # edges cutting
        new_corners[new_corners < 0] = 0
        new_corners[new_corners[:, 0] > w, 0] = w
        new_corners[new_corners[:, 1] > h, 1] = h
        corner_order = np.array([2, 3, 1, 0])  # contour draw order
        result = np.zeros(other.img.shape, np.uint8)
        cv.fillPoly(result, [new_corners[corner_order, :]], (255, 255, 255))
        return cv.bitwise_and(img, img, mask=result)

    def align_to(self, other, keep_percent: float = KEEP_PERCENT, img: np.array = None) -> np.array:
        """Align the image of this object with the image represented by the given other AlignHelper.

        Args:
            other (AlignHelper): The AlignHelper represents the image to align with.
            keep_percent (float, optional): The percent of matching that are allowed to use for alignment. Defaults to KEEP_PERCENT.
            img (np.array, optional): Another image to execute the alignment. If None execute the alignment on the image of this object. Defaults to None.

        Returns:
            np.array: An aligned image
        """
        h = AlignHelper.get_homography(self, other, keep_percent=keep_percent)
        if img is None:
            img = self.img
        if h is None:
            return img, None

        height, width = other.img.shape
        return cv.warpPerspective(img, h, (width, height)), h


# ==================#
# Images Denoising #
# ==================#

DENOISING_FILTER_STRENGTH = 3.0


def denoise(inspected_images: List[np.array], reference_images: List[np.array],
            filter_strength=DENOISING_FILTER_STRENGTH) -> List[np.array]:
    denoised_ins, denoised_ref = [], []
    for ins, ref in zip(inspected_images, reference_images):
        denoised_ins.append(
            cv.fastNlMeansDenoisingMulti([ins, ref], imgToDenoiseIndex=0, temporalWindowSize=1, h=[filter_strength]))
        denoised_ref.append(
            cv.fastNlMeansDenoisingMulti([ins, ref], imgToDenoiseIndex=1, temporalWindowSize=1, h=[filter_strength]))
    return denoised_ins, denoised_ref


def quantize(img, k=3):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(np.float32(img.reshape(-1, 2)), k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


class SSIM_helper:
    """An object to calculate and analyze SSIM method outputs.
    """
    MIN_SSIM_WIN_SIZE = 100

    def __init__(self, ins_img, ref_img):
        (self.score, self.similarity_diff) = skimage.metrics.structural_similarity(ins_img, ref_img, full=True)
        self.similarity_diff = (self.similarity_diff * 255).astype("uint8")
        self.thresh = cv.threshold(self.similarity_diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        contours = cv.findContours(self.thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours[0] if len(contours) == 2 else contours[1]
        self.ins = ins_img
        self.ref = ref_img

    def draw(self, bgr_background_img: np.array = None, contour: bool = True, xy_offset: Tuple[int] = (0, 0),
             color: Tuple[int] = (255, 0, 0)):
        """Draw the SSIM contours (pixels with high SSIM similarity difference) calculated for this object.

        Args:
            bgr_background_img (np.array, optional): An image to display the contours on. If None, it'll display on the self.ins image.Defaults to None.
            contour (bool, optional): If False it'll display the minimum rectangle around the contour, otherwise the contours. Defaults to True.
            xy_offset (tuple, optional): offset to shift the contout according to. Defaults to (0,0).
            color (tuple, optional): The color of the contours. Defaults to (255,0,0).

        Returns:
            np.array: Image with the SSIM contours.
        """
        if bgr_background_img is None:
            bgr_background_img = cv.cvtColor(self.ins, cv.COLOR_GRAY2BGR)
        for c in self.contours:
            area = cv.contourArea(c)
            c += xy_offset

            if contour:
                cv.drawContours(bgr_background_img, [c], 0, color, -1)
            else:
                x, y, w, h = cv.boundingRect(c)
                rgb_ins = cv.rectangle(bgr_background_img, (x, y), (x + w, y + h), color, 2)
        return bgr_background_img

    def filter_pts_by_contour(self, points: np.array) -> np.array:
        """Removes points that not inside any contours calculated for this object

        Args:
            points (np.array): to filter

        Returns:
            np.array: the filtered points array
        """
        filtered_points = []
        for p in points:
            for c in self.contours:
                if cv.pointPolygonTest(c, p, False) >= 0:
                    filtered_points.append(p)
        return filtered_points

    def filter_contour_by_points(self, points: np.array) -> None:
        """Filter the pixels with high SSIM difference and set them to 0, if their contour doesn't contain any points from the given list. 
        This function updates the contours of this object.

        Args:
            points (np.array): a list of points to filter with
        """
        filtered_contours = []
        cimg = np.zeros_like(self.ins)
        for c in self.contours:
            has_point = False
            points_inside = [p for p in points if cv.pointPolygonTest(c, p, False) >= 0]
            if len(points_inside) > 0:
                cv.drawContours(cimg, [c], -1, color=255, thickness=-1)
                corners = get_patch_edges_corners(np.array(points_inside))
                min_x = np.min(corners[:, 1])
                max_x = np.max(corners[:, 1])
                min_y = np.min(corners[:, 0])
                max_y = np.max(corners[:, 0])
        contours = cv.findContours(cimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours[0] if len(contours) == 2 else contours[1]

        return self.contours

    def _enlarge_rect_size(self, x: int, y: int, w: int, h: int, min_size: int) -> Tuple:
        """Given a window params, ensure that they represents a window with height and width that are at least equal to the given min size, otherwise they make the window larger.

        Args:
            x (int): The window start column index
            y (int): The window start row index
            w (int): The number of columns
            h (int): The number of rows.
            min_size (int): The minimum height and width of the given window

        Returns:
            Tuple: The larger window params
        """
        # TODO handle cases of small windows in the other edges
        if w < min_size:
            w_diff = min_size - w
            x = max(0, x - int(np.floor(w_diff / 2)))
            w = min_size
        if h < min_size:
            old = h
            h_diff = min_size - h
            y = max(0, y - int(np.floor(h_diff / 2)))
            h = min_size
        return x, y, w, h

    def get_contour_crops(self, xy_offset=(0, 0), min_crop_size: int = None) -> Tuple:
        """Crops of images from the ins and ref images of this object that contains contour area - for each contour a different crop.

        Args:
            xy_offset (tuple, optional): offset to shift the contour according to. Defaults to (0,0).
            min_crop_size (int, optional): minimum crop size-if a crop is smaller it'll enlarge it. Defaults to None.

        Returns:
            Tuple: ins_crops, ref_crops, contour_boundaries, contours
        """
        ins_crops, ref_crops, contour_boundaries, contours = [], [], [], []
        for c in self.contours:
            area = cv.contourArea(c)
            c += xy_offset
            if area > 40:
                x, y, w, h = cv.boundingRect(c)
                if min_crop_size is not None:
                    x, y, w, h = self._enlarge_rect_size(x, y, w, h, min_crop_size)
                y_end, x_end = min(y + h, self.ins.shape[0]), max(x + w, self.ins.shape[1])
                ins_crop = self.ins[y:y_end, x:x_end]
                ref_crop = self.ref[y:y_end, x:x_end]
                ins_crops.append(ins_crop)
                ref_crops.append(ref_crop)
                contour_boundaries.append((y, y_end, x, x_end))
                contours.append(c)
        return ins_crops, ref_crops, contour_boundaries, contours

    def fix_by_alignments(self, move_max: int = 40, min_crop_size: int = None):
        """Fix the alignment by a small linear shifing. The method is looking to maximize the corresponding crops correlation.
        The function Update the object's contours and threshold map.

        Args:
            move_max (int, optional): The maximakl shifting to consider. Defaults to 40.
            min_crop_size (int, optional): If is not None, before fixing it'll enlarge the crops to the minimum size. Defaults to None.

        Returns:
            Tuple: ins_crops, ref_crops, contour_boundaries, best_shifts.
        """

        ins_crops, ref_crops, contour_boundaries, contours = self.get_contour_crops(min_crop_size=min_crop_size)
        new_similarity_diff = np.zeros(self.similarity_diff.shape).astype("uint8")
        best_shifts = []

        for ins_crop, ref_crop, boundaries, c in zip(ins_crops, ref_crops, contour_boundaries, contours):
            ins_mask = cv.threshold(ins_crop, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            ref_mask = cv.threshold(ref_crop, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

            best_score = -np.inf
            best_aligned = None
            best_shift = [(0, 0)]
            shift_iter = ImageShiftingHelper.get_shift(ref_mask, move_max, move_max)
            for shifted_image, shift_values in shift_iter:
                score = np.sum(np.multiply(shifted_image, ins_mask))
                if score > best_score:
                    best_score = score
                    best_aligned = shifted_image
                    best_shift = shift_values

            diff = np.abs(ins_mask - best_aligned)

            contour_mask = np.zeros(new_similarity_diff.shape)
            cv.drawContours(contour_mask, [c], -1, 255, -1)
            y, y_end, x, x_end = boundaries
            new_similarity_diff[y:y_end, x:x_end] += np.logical_and(contour_mask[y:y_end, x:x_end], (diff > 0))
            new_similarity_diff = np.clip(new_similarity_diff, 0, 1).astype("uint8")
            best_shifts.append(best_shift)

        self.thresh = ~cv.threshold(new_similarity_diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        contours = cv.findContours(self.thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours[0] if len(contours) == 2 else contours[1]
        return ins_crops, ref_crops, contour_boundaries, best_shifts

    def scored_map(self, patch_size: int = 10) -> np.array:
        """A new scores map for the object ins and ref image.
        For each patch I replace the patch in the ins image with the corresponding ref image and 
        measure how mach the patch influence the ssim score.

        """
        y, x = self.ins.shape

        output_score_maps = np.zeros(self.ins.shape)
        r_c_pairs = np.array(np.meshgrid(np.arange(0, x, patch_size), np.arange(0, y, patch_size))).T.reshape(-1, 2)

        for i in range(r_c_pairs.shape[0]):
            rs, cs = r_c_pairs[i]
            re, ce = min(rs + patch_size, y), min(cs + patch_size, x)

            new_ins = np.copy(self.ins)
            new_ins[rs:re, cs:ce] = self.ref[rs:re, cs:ce]
            score = SSIM_helper(new_ins, self.ref).score
            output_score_maps[rs:re, cs:ce] = score

        return output_score_maps


class BolbDetector:
    def __init__(self, method_name='gaussian_diff', max_sigma=30, threshold=.05):
        methods = {'laplacian': lambda img: skimage.feature.blob_log(img, max_sigma=max_sigma, num_sigma=10,
                                                                     threshold=threshold), \
                   'gaussian_diff': lambda img: skimage.feature.blob_dog(img, max_sigma=max_sigma, threshold=threshold),
                   'hessian_det': lambda img: skimage.feature.blob_doh(img, max_sigma=max_sigma, threshold=0.1)}

        self.method = methods[method_name]

    def get_bolbs(self, image_gray):
        return self.method(image_gray)

    def draw_bolbs(self, image_gray):
        bolbs = self.get_bolbs(image_gray)
        bgr_img = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)
        color = (255, 0, 0)

        for bolb in bolbs:
            y, x, r = bolb

            cv.circle(bgr_img, (int(x), int(y)), 10, color=color)

        return bgr_img


class ImageShiftingHelper:
    """Helper object for images shifting
    """

    @staticmethod
    def get_shift(to_shift_img: np.array, rows_max_shift: int, columns_max_shift: int, rows_steps: int = 1,
                  cols_steps: int = 1) -> Generator:
        """Return Iterator on all the shifting cases.
        """

        for dy in range(-rows_max_shift, rows_max_shift, rows_steps):
            for dx in range(-columns_max_shift, columns_max_shift, cols_steps):
                shifted_image = ImageShiftingHelper.shift_by_params(to_shift_img, dy, dx)

                yield shifted_image, (dy, dx)

    def shift_by_params(to_shift_img: np.array, dy: int, dx: int) -> np.array:
        """Shifts the given image according to the given shifting values.
        """
        get_idx_start = lambda d, max_length: min(max(0, d), max_length)
        get_idx_end = lambda d, max_length: max(min(max_length, max_length + d), 0)
        shifted_image = np.zeros(to_shift_img.shape)
        ys, ye, xs, xe = get_idx_start(-dy, shifted_image.shape[0]), get_idx_end(-dy,
                                                                                 shifted_image.shape[0]), get_idx_start(
            -dx, shifted_image.shape[1]), get_idx_end(-dx, shifted_image.shape[1])
        ws, we, zs, ze = get_idx_start(dy, shifted_image.shape[0]), get_idx_end(dy,
                                                                                shifted_image.shape[0]), get_idx_start(
            dx, shifted_image.shape[1]), get_idx_end(dx, shifted_image.shape[1])
        shifted_image[ys:ye, xs:xe] = to_shift_img[ws:we, zs:ze]
        return shifted_image


# =================#
# Images Pyramids #
# =================#

def get_gaussian_pyramid(im: np.array, depth: int = 6):
    gaussian_layer = im.copy()
    rows, cols = im.shape
    pyramid = [gaussian_layer]
    for i in range(depth):
        cols //= 2
        rows //= 2
        gaussian_layer = cv.pyrDown(gaussian_layer, dstsize=(cols, rows))
        pyramid.append(gaussian_layer)

    return pyramid


def get_laplacian_pyramid(im: np.array, depth: int = 6):
    gp = get_gaussian_pyramid(im, depth)
    pyramid = [gp[-1]]
    cols, rows = gp[-1].shape
    for i in range(len(gp) - 1, 0, -1):
        cols *= 2
        rows *= 2
        upsampled = cv.pyrUp(gp[i], dstsize=(cols, rows))
        laplacian_layer = cv.subtract(gp[i - 1], upsampled)
        pyramid.append(laplacian_layer)
    return pyramid


def get_cliques(points):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric='euclidean').fit(points)
    distances, indices = nbrs.kneighbors(points)
    G = nx.Graph()
    for edge, dist in zip(indices, distances[:, 1]):
        G.add_edge(tuple(points[edge[0]]), tuple(points[edge[1]]), weight=dist)

    return list(nx.connected_components(G))


def get_patch_edges_corners(img_points: np.array):
    corners = np.zeros((4, 2), dtype=np.int)
    s = img_points.sum(axis=1)
    corners[0] = img_points[np.argmin(s)]
    corners[2] = img_points[np.argmax(s)]

    diff = np.diff(img_points, axis=1)
    corners[1] = img_points[np.argmin(diff)]
    corners[3] = img_points[np.argmax(diff)]

    return corners
