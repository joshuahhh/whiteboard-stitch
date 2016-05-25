import multiprocessing.dummy
from itertools import repeat

import numpy as np
from numba import jit
import cv2

from spimage import Image, operate, composite, blend_subimages, sum_subimages
from sphomography import find_features, find_homography, apply_homography, apply_homography_tight
from spvoronoi import voronoi, clip
from profile import profile


@jit
def stable_divide(num, denom):
    return (num + 4.) / (denom + 4)


def blur_with_mask(image, mask, blur_op):
    image_cropped_to_mask = operate(
        lambda image, mask: image * (
            mask
            if len(image.shape) == 2
            else mask[:, :, np.newaxis]),
        image, mask)
    image_blurred = blur_op(image_cropped_to_mask.astype(np.float32))
    mask_blurred = blur_op(mask.astype(np.float32))
    image_blurred.array /= (
        mask_blurred.array
        if len(image_blurred.array.shape) == 2
        else mask_blurred.array[:, :, np.newaxis]
    )
    return image_blurred


def detail_transfer(background_blurred, foreground, mask, blur_op):
    foreground_blurred = blur_with_mask(foreground, mask, blur_op)

    detail_array = stable_divide(foreground.array, foreground_blurred.array)
    if mask is not None:
        detail_array *= mask.array[:, :, np.newaxis]

    detail_array *= background_blurred.array

    return Image(detail_array, foreground.system)


class StitchingJob:
    def __init__(self, establishing, closes, xcloses=None):
        self.establishing = establishing
        self.closes = closes
        self.xcloses = xcloses or []

    def find_homographies(self, downsample_scale=0.5, num_threads=None):
        with profile():
            print 'find_homographies'
            # This uses a thread pool (a dummy multiprocessing pool) to pipeline the
            # OpenCV work and take better advantage of multicore machines.
            pool = multiprocessing.dummy.Pool(processes=num_threads)

            self.e_features = find_features(self.establishing)
            c_features_results = [
                pool.apply_async(
                    lambda c: find_features(
                        c.resize(scale=downsample_scale) if downsample_scale < 1 else c),
                    (c,))
                for c in self.closes]
            self.c_homs = pool.map(lambda c_features_result:
                find_homography(c_features_result.get(), self.e_features),
                c_features_results)
            self.c_features = [
                c_features_result.get()
                for c_features_result in c_features_results]

    def calculate_areas(self):
        # These areas are in establishing-shot space.
        self.areas = [
            cv2.contourArea(np.array([h(pt).coords for pt in c.corners()]))
            for (c, h) in zip(self.closes, self.c_homs)
        ]

    def calculate_canvas_scale(self):
        # OK SO: For a given close-shot, we examine how much the homography
        # shrinks it. The maximum (linear) shrinkage determines how much we
        # should scale the canvas.
        scales = [
            np.sqrt(close.area() / area)
            for close, area in zip(self.closes, self.areas)
        ]
        self.canvas_scale = min(max(scales), 3)  # TODO: override the max of 3?
        print 'canvas scale is %f (from %s)' % (self.canvas_scale, str(scales))

    def generate_masks_voronoi(self):
        with profile():
            print 'generate_masks_voronoi'

            centers = [c_hom(close.center()) for c_hom, close in zip(self.c_homs, self.closes)]
            self.c_voronoi_facets = voronoi(
                centers, self.establishing.system, self.establishing.dims)

            mask_template = (
                self.establishing
                .resize(scale=self.canvas_scale)
                .pipe(lambda x: np.zeros(x.shape[:2]))
            )
            homography_masks = [
                mask_template.fill_poly(
                    clip(map(c_hom, close.corners()), mask_template.corners()),
                    color=1).erode(3)
                for c_hom, close in zip(self.c_homs, self.closes)
            ]
            self.c_masks = [
                operate(lambda x, y: x * y,
                    mask_template.fill_poly(clip(facet, mask_template.corners()), color=1),
                    homography_mask)
                for facet, homography_mask in zip(self.c_voronoi_facets, homography_masks)
            ]

    def generate_masks_stacked(self):
        with profile():
            print 'generate_masks_stacked'

            self.calculate_areas()
            area_order = sorted(range(len(self.areas)), key=lambda i: self.areas[i])

            mask_template = (
                self.establishing
                .resize(scale=self.canvas_scale)
                .pipe(lambda x: np.zeros(x.shape[:2]))
            )
            available = mask_template.pipe(lambda x: x + 1)
            self.c_masks = [None] * len(self.closes)
            for i in area_order:
                c_hom, close = self.c_homs[i], self.closes[i]
                homography_boundary = map(c_hom, close.corners())
                clipped_homography_boundary = clip(homography_boundary, mask_template.corners())
                c_mask = (
                    mask_template
                    .fill_poly(clipped_homography_boundary, color=1)
                    .erode(3))
                c_mask.array *= available.array
                c_mask = c_mask
                self.c_masks[i] = c_mask

                available.array -= c_mask.array

    def simple_stitch(self):
        with profile():
            print 'simple_stitch'

            canvas = self.establishing.resize(scale=self.canvas_scale)

            def process(inputs):
                close, c_hom = inputs
                foreground, homography_mask = apply_homography(
                    close, c_hom, canvas.system, canvas.dims)
                print 'processed'
                return foreground

            pool = multiprocessing.dummy.Pool()
            self.simple_stitch_outputs = pool.map(
                process, zip(self.closes, self.c_homs))

            for foreground, mask in zip(self.simple_stitch_outputs, self.c_masks):
                composite(canvas, foreground, mask, inplace=True)

            self.simple_stitch_output = canvas
            return self.simple_stitch_output

    def detail_transfer_stitch_pt_1(self, detail_transfer_blur_op,
            edge_blend_radius):
        with profile():
            print 'detail_transfer_stitch_pt_1'

            canvas = (
                self.establishing
                .white_balance()
                .normalize()
                .resize(scale=self.canvas_scale)
            )
            canvas_blurred = detail_transfer_blur_op(canvas.astype(np.float32))
            # TODO: not being used
            # pool = multiprocessing.dummy.Pool()
            self.detail_transfer_stitch_outputs = map(
                self._detail_transfer_stitch_step,
                zip(repeat((canvas, canvas_blurred, detail_transfer_blur_op, edge_blend_radius)),
                    self.closes, self.c_homs, self.c_masks))
            print 'all details are transferred'

    def _detail_transfer_stitch_step(self, inputs):
        ((canvas, canvas_blurred, detail_transfer_blur_op, edge_blend_radius),
            close, c_hom, c_mask) = inputs
        foreground, homography_mask = apply_homography_tight(
            close, c_hom, canvas.system, canvas.dims, margin=100)
        full_mask = c_mask.crop_like(homography_mask.system, homography_mask.dims)
        # Blur the mask, but only within the homography boundaries:
        full_mask_blurred = operate(lambda x, y: x * y,
            blur_with_mask(full_mask, homography_mask, lambda im: im.blur(edge_blend_radius)),
            homography_mask)

        transferred = detail_transfer(canvas_blurred.crop_like(foreground.system, foreground.dims),
                                      foreground,
                                      homography_mask, detail_transfer_blur_op)

        print 'processed'
        return (transferred, full_mask, full_mask_blurred)

    def detail_transfer_stitch_pt_2(self, detail_transfer_blur_op,
                                    edge_blend_radius):
        with profile():
            print 'detail_transfer_stitch_pt_2'

            canvas = (
                self.establishing
                .white_balance()
                .normalize()
                .resize(scale=self.canvas_scale)
            )
            images, full_masks, full_masks_blurred = zip(*self.detail_transfer_stitch_outputs)
            print 'computing outside mask'
            print 'step 1'
            outside_mask_blurred = sum_subimages(full_masks, canvas.system, canvas.dims)
            print 'step 2'
            outside_mask_blurred = outside_mask_blurred.pipe(lambda x: (x == 0).astype(float))
            print 'step 3'
            outside_mask_blurred = outside_mask_blurred.blur(edge_blend_radius)
            print outside_mask_blurred.array.shape
            print full_masks_blurred[0].array.shape
            del full_masks  # to save some memory
            print 'blending'
            self.detail_transfer_stitch_output = blend_subimages(
                (canvas,) + images,
                (outside_mask_blurred,) + full_masks_blurred,
                canvas.system,
                canvas.dims)
            return self.detail_transfer_stitch_output

    def draw_mask_boundaries_onto(self, canvas, color=(0, 255, 0), halfwidth=3):
        kernel = np.ones((2 * halfwidth + 1, 2 * halfwidth + 1))
        for mask in self.c_masks:
            resized_mask = mask.resize(width=canvas.width)
            # this next line is crazy-slow for large images?
            boundary = resized_mask.pipe(lambda arr: cv2.dilate(arr, kernel) - arr)
            canvas.array[boundary.array.astype(bool)] = color
        return canvas

    def draw_masks_onto(self, canvas, color=(0, 255, 0), opacity=0.2):
        for mask in self.c_masks:
            resized_mask = mask.resize(width=canvas.width)
            canvas.array[resized_mask.array.astype(bool)] = (
                np.array(color) * opacity +
                canvas.array[resized_mask.array.astype(bool)] * (1 - opacity))
        return canvas

    def draw_homography_boundaries_onto(self, canvas, color=(0, 255, 0), width=10):
        for c, h in zip(self.closes, self.c_homs):
            canvas.draw_polyline(map(h, c.corners()), color=color, width=width, inplace=True)
        return canvas

    def draw_voronoi_diagram_onto(self, canvas, color=(0, 255, 0), width=10):
        for facet in self.c_voroni_facets:
            canvas.draw_polyline(clip(facet, canvas.corners()),
                                 color=color, width=width, inplace=True)
        return canvas
