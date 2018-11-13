import os
import errno
import datetime

import cv2
import imutils
import numpy as np
from numba import jit

from IPython.core.display import display_png, display_html


temp_images_dir = 'temp-images'


def show_image(arr):
    resized = imutils.resize(arr, width=800)
    png_to_show = cv2.imencode('.png', resized)[1].tostring()
    display_png(png_to_show, raw=True)

    png_to_save = cv2.imencode('.png', arr)[1].tostring()
    try:
        os.makedirs(temp_images_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    file_name = datetime.datetime.now().isoformat().replace(':', '_') + '.png'
    file_path = os.path.join(temp_images_dir, file_name)
    with open(file_path, 'w') as f:
        f.write(png_to_save)

    url = 'http://localhost:8888/files/%s' % file_path
    display_html('<a href="%s" target="_blank">%s</a>' % (url, file_name), raw=True)


# [IMAGE] SPACE: world a given image lives in (non-numeric, pretty abstract)
# [COORDINATE] SYSTEM: mapping from coords to point in image space
#
# In theory, a space has no canonical systems.
# In practice, we implement everything as follows:
#   A space is just a wrapper around an optional label. (If the label is
#     provided, it determines space identity.)
#   A system is just a space together with an affine matrix A so that the point
#     corresponding to (x, y) is A(x, y) in the original image. So the system of
#     a downsampled image will have an affine matrix which says "scale up the
#     point!".
#   An image is an image array together with a system.

class ImageSpace:
    def __init__(self, label=None):
        self.label = label

    def __eq__(self, other):
        return (
            self.__class__.__name__ == other.__class__.__name__ and
            (self.label == other.label if self.label else self is other)
        )


class CoordSystem:
    # matrix takes you FROM system coords TO base coords
    def __init__(self, space, matrix=None):
        assert space.__class__.__name__ == 'ImageSpace', space
        self.space = space
        if matrix is not None:
            assert matrix.__class__.__name__ == np.ndarray.__name__, matrix
            self.matrix = matrix
            self.matrix_inv = np.linalg.inv(matrix)
        else:
            self.matrix = np.eye(3)
            self.matrix_inv = np.eye(3)

    def translate(self, new_origin):
        translation = np.array([[1, 0, new_origin[0]], [0, 1, new_origin[1]], [0, 0, 1]])
        new_matrix = self.matrix.dot(translation)
        return CoordSystem(self.space, new_matrix)

    def translation_origin(self, other_system):
        assert self.space == other_system.space
        self_coords_to_other_coords = other_system.matrix_inv.dot(self.matrix)
        # Make sure it's a translation of the other system:
        assert (self_coords_to_other_coords[:3, :2] == np.array([[1, 0], [0, 1], [0, 0]])).all(), (
            self_coords_to_other_coords)
        other_coords_of_self_origin = self_coords_to_other_coords[:2, 2]
        return other_coords_of_self_origin

    def rescale(self, factor):
        new_matrix = self.matrix.dot(np.diag([factor, factor, 1]))
        return CoordSystem(self.space, new_matrix)

    def to_base_coords(self, coords):
        return self.matrix.dot(np.append(coords, 1))[:-1]

    def from_base_coords(self, coords):
        return self.matrix_inv.dot(np.append(coords, 1))[:-1]

    def same_space(self, other_system):
        return self.space == other_system.space

    def __eq__(self, other):
        return (
            other.__class__.__name__ == self.__class__.__name__ and
            self.space == other.space and
            np.array_equal(self.matrix, other.matrix)
        )


class ImagePoint:
    # coords can be a list or a 1D numpy array
    def __init__(self, coords, system):
        assert len(coords) == 2, coords
        self.coords = coords
        assert system.__class__.__name__ == 'CoordSystem', system
        self.system = system

    def __repr__(self):
        return '<ImagePoint %s in CoordSystem %s>' % (repr(self.coords), hex(id(self.system)))

    def in_system(self, other_system):
        if self.system == other_system:
            return self
        assert self.system.same_space(other_system), (
            'Cannot interpret point (in %s) in %s!' % (repr(self.system), repr(other_system))
            )
        base_coords = self.system.to_base_coords(self.coords)
        return ImagePoint(other_system.from_base_coords(base_coords), other_system)

    def round_coords(self):
        return tuple(int(round(c)) for c in self.coords)


def corners(system, dims):
    w, h = dims
    return [ImagePoint(c, system) for c in [(0, 0), (w, 0), (w, h), (0, h)]]


class Image:
    def __init__(self, array, system):
        # `array` shold be either
        #   height x width x 3 (for three-channel) or
        #   height x width (for one-channel)
        assert array.__class__.__name__ == np.ndarray.__name__, array
        assert len(array.shape) in (2, 3), array
        self.array = array
        assert system.__class__.__name__ == 'CoordSystem', system
        self.system = system

    @staticmethod
    def from_array(array, label=None):
        space = ImageSpace(label)
        system = CoordSystem(space)
        image = Image(array, system)
        return image

    @staticmethod
    def from_file(name):
        array = cv2.imread(name)
        label = os.path.abspath(name)
        return Image.from_array(array, label)

    @property
    def width(self):
        return self.array.shape[1]

    @property
    def height(self):
        return self.array.shape[0]

    @property
    def dims(self):
        return self.array.shape[1::-1]

    def area(self):
        return self.width * self.height

    def resize(self, width=None, scale=None):
        if (not width) and scale:
            width = int(self.width * scale)
        resized_array = imutils.resize(self.array, width)
        new_system = self.system.rescale(float(self.width) / resized_array.shape[1])
        return Image(resized_array, new_system)

    def show(self):
        if self.array.max() <= 1:
            show_image(255 * self.array)
        else:
            show_image(self.array)

    def save(self, name):
        cv2.imwrite(name, self.array)

    def __repr__(self):
        space_label = self.system.space.label
        maybe_label = (" " + space_label) if space_label else ""
        return "<Image (%i x %i)" % (self.width, self.height) + maybe_label + ">"

    def corners(self):
        return corners(self.system, self.dims)

    def center(self):
        return ImagePoint([self.width / 2, self.height / 2], self.system)

    def __getitem__(self, key):
        assert len(key) == 2, 'Indexing on an Image requires exactly two arguments'
        x_key, y_key = key

        # This is either cropping or lookup...
        getitem_array = self.array.__getitem__((y_key, x_key))
        if len(getitem_array.shape) > 1:
            # There's probably an easier way to do this... (and no strides, please)
            x_translate = range(self.width)[x_key][0]
            y_translate = range(self.height)[y_key][0]
            cropped_system = self.system.translate([x_translate, y_translate])
            return Image(getitem_array, cropped_system)
        else:
            return getitem_array

    def draw_line(self, pt1, pt2, color=(0, 255, 0), width=1, inplace=False):
        array_copy = self.array.copy() if not inplace else self.array
        pt1_here = pt1.in_system(self.system)
        pt2_here = pt2.in_system(self.system)
        cv2.line(array_copy, pt1_here.round_coords(), pt2_here.round_coords(), color, width)
        return Image(array_copy, self.system)

    def draw_polyline(self, pts, closed=True, color=(0, 255, 0), width=1, inplace=False):
        array_copy = self.array.copy() if not inplace else self.array
        pts_here = [pt.in_system(self.system) for pt in pts]
        cv2.polylines(array_copy, np.array([[pt.round_coords() for pt in pts_here]]),
                      closed, color, width)
        return Image(array_copy, self.system)

    def draw_circle(self, center, radius, color=(0, 255, 255), width=-1, inplace=False):
        array_copy = self.array.copy() if not inplace else self.array
        center_here = center.in_system(self.system)
        cv2.circle(array_copy, center_here.round_coords(), radius, color, width)
        return Image(array_copy, self.system)

    def fill_poly(self, points, color, inplace=False):
        array_copy = self.array.copy() if not inplace else self.array
        pts_here = [pt.in_system(self.system) for pt in points]
        cv2.fillPoly(array_copy, np.array([[pt.round_coords() for pt in pts_here]]), color)
        return Image(array_copy, self.system)

    def copy(self):
        return Image(self.array.copy(), self.system)

    # It's assumed that "func" takes an image array and returns an image array in the same system
    # (e.g., an OpenCV filter)
    def pipe(self, func, *args, **kwargs):
        return Image(func(self.array, *args, **kwargs), self.system)

    def astype(self, typ):
        return self.pipe(lambda array: array.astype(typ))

    def normalize(self):
        return self.pipe(lambda array: np.clip(array, 0, 255).astype(np.uint8))

    def blur(self, radius):
        return self.pipe(cv2.blur, (2 * radius + 1,) * 2)

    def bilateral_blur(self, radius, sigma_color, sigma_space):
        return self.pipe(cv2.bilateralFilter, d=2 * radius + 1,
                         sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def erode(self, radius):
        return self.pipe(cv2.erode, np.ones((2 * radius + 1,) * 2, np.uint8))

    def white_balance(self):
        return self.pipe(lambda a: a / find_white(a.astype(np.uint8)) * [255, 255, 255])

    def crop_like(self, other_system, other_dims):
        top_left = other_system.translation_origin(self.system)
        assert (top_left >= 0).all() and np.allclose(top_left, top_left.round()), top_left
        top_left = top_left.round().astype(int)
        bottom_right = top_left + other_dims
        assert (bottom_right <= self.dims).all(), (bottom_right, self.dims)
        return self[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


def operate(func, *images):
    for image in images[1:]:
        assert image.system == images[0].system
        assert image.dims == images[0].dims
#     output_array = func(*(image.array.astype(float) for image in images)).astype(np.uint8)
    output_array = func(*(image.array for image in images))
    return Image(output_array, images[0].system)


def find_white(array):
    pixels = array.reshape((-1, 3))
    grey = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).reshape(-1)
    gsort = grey.argsort()
    top_5_percent = gsort[-len(gsort) / 20:]
    return pixels[top_5_percent].mean(axis=0)


class ImageFunction:
    def __init__(self, src_system, dst_system, coord_function=None):
        assert src_system.__class__.__name__ == 'CoordSystem'
        self.src_system = src_system
        assert dst_system.__class__.__name__ == 'CoordSystem'
        self.dst_system = dst_system
        if coord_function:
            assert hasattr(coord_function, '__call__')
            self.coord_function = coord_function

    def __call__(self, point):
        src_point = point.in_system(self.src_system)
        dst_coords = self.coord_function(src_point.coords)
        return ImagePoint(dst_coords, self.dst_system)


# Returns (output_image, embedding_left, embedding_right)
def abut(image1, image2, embeddings=False):
    (w1, h1) = image1.dims
    (w2, h2) = image2.dims
    output_array = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
    output_array[0:h1, 0:w1] = image1.array
    output_array[0:h2, w1:] = image2.array
    output_image = Image.from_array(output_array)
    if embeddings:
        embedding_left = ImageFunction(image1.system, output_image.system,
                                       lambda (x, y): (x, y))
        embedding_right = ImageFunction(image2.system, output_image.system,
                                        lambda (x, y): (x + w1, y))
        return (output_image, embedding_left, embedding_right)
    else:
        return output_image


def composite(background_image, foreground_image, mask, inplace=False):
    assert background_image.system == foreground_image.system
    assert background_image.system == mask.system

    background_copy = background_image.copy() if not inplace else background_image
    background_copy.array[mask.array == 1] = foreground_image.array[mask.array == 1]
    return background_copy


@jit(nopython=True, cache=True)
def sum_arrs_with_offsets(shape, arrs, offsets):
    base = np.zeros(shape)
    for arr, offset in zip(arrs, offsets):
        base[offset[1]:offset[1] + arr.shape[0], offset[0]:offset[0] + arr.shape[1]] += arr
    return base


@jit(cache=True)
def blend_arrs_with_offsets(arr_shape, arrs, masks, offsets):
    '''Must receive no NaNs! That's what masks are for.'''
    mask_sum = sum_arrs_with_offsets(arr_shape[:2], masks, offsets)

    base = np.zeros(arr_shape)
    for arr, mask, offset in zip(arrs, masks, offsets):
        mask_sum_cropped = (
            mask_sum[offset[1]:offset[1] + arr.shape[0], offset[0]:offset[0] + arr.shape[1]]
        )
        base[offset[1]:offset[1] + arr.shape[0], offset[0]:offset[0] + arr.shape[1]] += (
            arr * np.true_divide(mask, mask_sum_cropped)[:, :, np.newaxis]
        )
    return base


@jit(cache=True)
def nan_to_zero(arr, inplace=False):
    '''numpy's "nan_to_num" is crazy-slow, and doesn't have an inplace option.'''
    copy = arr if inplace else arr.copy()
    copy[np.isnan(copy)] = 0
    return copy


def blend_subimages(images, masks, output_system, output_dims):
    # Make sure masks are aligned with images
    assert all(image.system == mask.system for image, mask in zip(images, masks))

    offsets = tuple(image.system.translation_origin(output_system).astype(int)
                    for image in images)

    arr = blend_arrs_with_offsets(
        (output_dims[1], output_dims[0], 3),
        tuple(nan_to_zero(image.array).astype(np.float64) for image in images),
        tuple(nan_to_zero(mask.array).astype(np.float64) for mask in masks),
        offsets)

    return Image(arr, output_system)


def sum_subimages(images, output_system, output_dims):
    offsets = tuple(image.system.translation_origin(output_system).astype(int)
                    for image in images)

    shape = (
        (output_dims[1], output_dims[0], 3)
        if len(images[0].array.shape) > 2
        else (output_dims[1], output_dims[0]))
    arr = sum_arrs_with_offsets(
        shape,
        tuple(image.array for image in images),
        offsets)

    return Image(arr, output_system)
