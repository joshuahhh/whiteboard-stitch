from collections import namedtuple
import cv2
import numpy as np

from spimage import Image, ImagePoint, ImageFunction, corners


Feature = namedtuple('Feature', ['point', 'descriptor', 'source_image'])


class Homography(ImageFunction):
    def __init__(self, src_system, dst_system, matrix, inlier_matches=None):
        self.matrix = matrix
        self.inlier_matches = inlier_matches
        ImageFunction.__init__(self, src_system, dst_system)

    def coord_function(self, coords):
        coords_array = np.array([[coords]], dtype='float32')
        return cv2.perspectiveTransform(coords_array, self.matrix)[0][0]

    def set_src_system(self, new_src_system):
        if new_src_system == self.src_system:
            return self
        assert self.src_system.same_space(new_src_system)
        new_matrix = self.matrix.dot(self.src_system.matrix_inv).dot(new_src_system.matrix)
        return Homography(new_src_system, self.dst_system, new_matrix, self.inlier_matches)

    def set_dst_system(self, new_dst_system):
        if new_dst_system == self.dst_system:
            return self
        assert self.dst_system.same_space(new_dst_system)
        new_matrix = new_dst_system.matrix_inv.dot(self.dst_system.matrix).dot(self.matrix)
        return Homography(self.src_system, new_dst_system, new_matrix, self.inlier_matches)

    def invert(self):
        return Homography(self.dst_system, self.src_system, np.linalg.inv(self.matrix))


def find_features(image):
    detector = cv2.xfeatures2d.SIFT_create()
    kps, descriptors = detector.detectAndCompute(image.array, None)
    return [Feature(point=ImagePoint(kp.pt, image.system),
                    descriptor=descriptor, source_image=image)
            for (kp, descriptor) in zip(kps, descriptors)]


def find_homography(features1, features2, ratio=0.75, reproj_thresh=4.0):
    descriptors1 = np.array([f.descriptor for f in features1])
    descriptors2 = np.array([f.descriptor for f in features2])

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    matches = []
    for m in raw_matches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].queryIdx, m[0].trainIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        system1 = features1[0].point.system
        system2 = features2[0].point.system

        coords1 = np.float32([features1[i].point.in_system(system1).coords
                              for (i, _) in matches])
        coords2 = np.float32([features2[i].point.in_system(system2).coords
                              for (_, i) in matches])

        # compute the homography between the two sets of points
        (H, mask) = cv2.findHomography(coords1, coords2, cv2.RANSAC, reproj_thresh)

        inlier_matches = [match for (match, status) in zip(matches, mask) if status == 1]

        return Homography(src_system=system1, dst_system=system2, matrix=H,
                          inlier_matches=inlier_matches)

    # otherwise, no homograpy could be computed
    return None


def homography_mask(src_system, src_dims, H, dst_system, dst_dims, erode=3):
    H_in_systems = H.set_src_system(src_system).set_dst_system(dst_system)
    src_corners = corners(src_system, src_dims)
    dst_corners = [H_in_systems(pt) for pt in src_corners]
    dst_coords = [pt.coords for pt in dst_corners]
    mask = np.zeros((dst_dims[1], dst_dims[0], 3))
    cv2.fillPoly(mask, np.array(dst_coords).reshape((1, -1, 2)).astype(np.int32), (255, 255, 255))
    if erode > 1:
        mask = cv2.erode(mask, np.ones((erode, erode)))
    return Image(mask[:, :, 0] / 255, dst_system)


def apply_homography(src_image, H, dst_system, dst_dims, erode=3):
    H_in_systems = H.set_src_system(src_image.system).set_dst_system(dst_system)
    dst_array = cv2.warpPerspective(src_image.array, H_in_systems.matrix, dst_dims)
    dst_image = Image(dst_array, dst_system)
    mask = homography_mask(src_image.system, src_image.dims, H, dst_system, dst_dims, erode)
    return dst_image, mask


def apply_homography_tight(src_image, H, dst_system, dst_dims, margin=100, erode=3):
    H_in_systems = H.set_src_system(src_image.system).set_dst_system(dst_system)

    src_corners = corners(src_image.system, src_image.dims)
    dst_corners = [H_in_systems(pt) for pt in src_corners]
    dst_coords = np.array([pt.coords for pt in dst_corners])
    top_left = np.floor(dst_coords.min(axis=0))
    bottom_right = np.ceil(dst_coords.max(axis=0))

    top_left_with_margin = np.maximum(top_left - margin, 0)
    bottom_right_with_margin = np.minimum(bottom_right + margin, dst_dims)

    tight_system = dst_system.translate(top_left_with_margin)
    tight_dims = (bottom_right_with_margin - top_left_with_margin).astype(int)

    return apply_homography(src_image, H, tight_system, tuple(tight_dims), erode)


def apply_homography_onto(src_image, H, dst_image, inplace=False):
    applied_image, mask = apply_homography(src_image, H, dst_image.system, dst_image.dims)
    dst_image_copy = dst_image.copy() if not inplace else dst_image
    dst_image_copy.array[mask.array == 1] = applied_image.array[mask.array == 1]
    return dst_image_copy
