import cv2
import numpy as np
import imutils
from collections import namedtuple

Descriptors = namedtuple('Descriptors', ['kps', 'descriptors', 'image'])
HomographyFromDescriptors = namedtuple('HomographyFromDescriptors', ['H', 'matches', 'descriptors1', 'descriptors2'])

def detect_and_describe(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect keypoints in the image
    detector = cv2.xfeatures2d.SIFT_create()
    kps, descriptors = detector.detectAndCompute(image, None)

#     # extract features from the image
#     extractor = cv2.DescriptorExtractor_create("SIFT")
#     kps, descriptors = extractor.compute(gray, kps)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    return Descriptors(kps, descriptors, image)

def match_keypoints(descriptors1, descriptors2, ratio, reproj_thresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(descriptors1.descriptors, descriptors2.descriptors, 2)
    matches = []

    # loop over the raw matches
    for m in raw_matches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        pts1 = np.float32([descriptors1.kps[i] for (_, i) in matches])
        pts2 = np.float32([descriptors2.kps[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, mask) = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)

        inlier_matches = [match for (match, status) in zip(matches, mask) if status == 1]

        return HomographyFromDescriptors(H, inlier_matches, descriptors1, descriptors2)

    # otherwise, no homograpy could be computed
    return None

def draw_matches(homography_from_descriptors):
    matches = homography_from_descriptors.matches
    descriptors1 = homography_from_descriptors.descriptors1
    descriptors2 = homography_from_descriptors.descriptors2
    image1 = descriptors1.image
    kps1 = descriptors1.kps
    image2 = descriptors2.image
    kps2 = descriptors2.kps

    # initialize the output visualization image
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = image1
    vis[0:h2, w1:] = image2

    for (trainIdx, queryIdx) in matches:
        pt1 = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
        pt2 = (int(kps2[trainIdx][0]) + w1, int(kps2[trainIdx][1]))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    # return the visualization
    return vis

def image_corners(image):
    (h, w) = image.shape[:2]
    return [(0, 0), (w, 0), (w, h), (0, h)]

def draw_rect(image1, image2, H):
    pts = cv2.perspectiveTransform(np.array([image_corners(image1)], dtype='float32'), H).reshape((1, -1, 2)).astype(np.int32)
    to_return = image2.copy()
    cv2.polylines(to_return, pts, True, (0, 255, 0), 2)
    return to_return


def scaling(x, y=None):
    if y == None:
        y = x
    return np.array([[x, 0, 0], [0, y, 0], [0, 0, 1]])

def better_warp_onto(image1, image2, H):
    warped = cv2.warpPerspective(image1, H, image2.shape[1::-1])

    pts = cv2.perspectiveTransform(np.array([image_corners(image1)], dtype='float32'), H).reshape((1, -1, 2)).astype(np.int32)
    mask = np.zeros_like(image2)
    cv2.fillPoly(mask, pts, (255, 255, 255))
    # mask = cv2.erode(mask, np.ones((3, 3)))

#     return mask * warped + (1 - mask) * image2
    center = (image2.shape[1] / 2, image2.shape[0] / 2)
    print imutils.resize(warped, width=100).shape
    print image2.shape
    print imutils.resize(mask, width=100).shape
    print center
    return cv2.seamlessClone(imutils.resize(warped, width=100), image2, imutils.resize(mask, width=100), center, cv2.NORMAL_CLONE)


imageA_raw = cv2.imread('2016-03-14 17.46.19.jpg')
imageB_raw = cv2.imread('establishing.jpg')


imageA = imutils.resize(imageA_raw, width=800)
imageB = imutils.resize(imageB_raw, width=800)


descriptorsA = detect_and_describe(imageA)
descriptorsB = detect_and_describe(imageB)

ratio = 0.75
reprojThresh = 4.0

homography_from_descriptors = match_keypoints(descriptorsA, descriptorsB, ratio, reprojThresh)

vis = draw_matches(homography_from_descriptors)

scaling_factor = 3
downsampling_factor = imageA_raw.shape[1] / 800
H_scaled = (
    # scaling(downsampling_factor)
    scaling(scaling_factor)
    .dot(homography_from_descriptors.H)
    # .dot(scaling(1.0/downsampling_factor))
)
imageB_scaled = imutils.resize(imageB, width=imageB.shape[1] * scaling_factor)

better_warp_onto(imageA, imageB_scaled, H_scaled)
