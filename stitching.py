import multiprocessing.dummy
from itertools import repeat

import numpy as np
from numba import jit

from spimage import Image, operate, composite, blend_with_offsets, sum_arrs_with_offsets
from sphomography import find_features, find_homography, apply_homography, apply_homography_tight
from spvoronoi import voronoi, clip
from profile import profile


@jit
def stable_divide(num, denom):
    return (num + 4.) / (denom + 4)


def blur_with_mask(image, mask, radius):
    image_blurred = image.astype(float).blur(radius)
    mask_blurred = mask.blur(radius)
    image_blurred.array /= (
        mask_blurred.array
        if len(image_blurred.array.shape) == 2
        else mask_blurred.array[:, :, np.newaxis]
    )
    return image_blurred


def detail_transfer(background_blurred, foreground, mask, radius):
    foreground_blurred = blur_with_mask(foreground, mask, radius)

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

    def find_homographies(self, downsample_scale=0.5):
        with profile():
            print 'find_homographies'
            # This uses a thread pool (a dummy multiprocessing pool) to pipeline the
            # OpenCV work and take better advantage of multicore machines.
            pool = multiprocessing.dummy.Pool()

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

    def find_voronoi(self):
        with profile():
            print 'find_voronoi'
            centers = [c_hom(close.center()) for c_hom, close in zip(self.c_homs, self.closes)]
            self.c_facets = voronoi(centers, self.establishing.system, self.establishing.dims)

    def simple_stitch(self, canvas_scale):
        with profile():
            print 'simple_stitch'

            canvas = self.establishing.resize(scale=canvas_scale)

            def process(inputs):
                close, c_hom, facet = inputs
                foreground, homography_mask = apply_homography(
                    close, c_hom, canvas.system, canvas.dims)
                voronoi_mask = (
                    canvas
                    .pipe(lambda x: np.zeros(x.shape[:2]))
                    .fill_poly(clip(facet, canvas.corners()), color=1)
                )
                full_mask = operate(lambda x, y: x * y, homography_mask, voronoi_mask)

                print 'processed'
                return (foreground, full_mask)

            pool = multiprocessing.dummy.Pool()
            self.simple_stitch_outputs = pool.map(
                process, zip(self.closes, self.c_homs, self.c_facets))

            for foreground, mask in self.simple_stitch_outputs:
                composite(canvas, foreground, mask, inplace=True)

            self.simple_stitch_output = canvas
            return self.simple_stitch_output

    def detail_transfer_stitch_pt_1(self, canvas_scale, detail_transfer_radius, edge_blend_radius):
        with profile():
            print 'detail_transfer_stitch_pt_1'

            canvas = self.establishing.white_balance().normalize().resize(scale=canvas_scale)
            canvas_blurred = canvas.blur(detail_transfer_radius)
            # TODO: not being used
            # pool = multiprocessing.dummy.Pool()
            self.detail_transfer_stitch_outputs = map(
                self._detail_transfer_stitch_step,
                zip(repeat((canvas, canvas_blurred, detail_transfer_radius, edge_blend_radius)),
                    self.closes, self.c_homs, self.c_facets))
            print 'all details are transferred'

    def _detail_transfer_stitch_step(self, inputs):
        ((canvas, canvas_blurred, detail_transfer_radius, edge_blend_radius),
            close, c_hom, facet) = inputs
        foreground, homography_mask = apply_homography_tight(
            close, c_hom, canvas.system, canvas.dims, margin=100)
        voronoi_mask = (
            homography_mask
            .pipe(lambda x: np.zeros(x.shape[:2]))
            .fill_poly(clip(facet, canvas.corners()), color=1, inplace=True)
        )
        full_mask = operate(lambda x, y: x * y, homography_mask, voronoi_mask)
        # Blur the mask, but only within the homography boundaries:
        full_mask_blurred = operate(lambda x, y: x * y,
            blur_with_mask(full_mask, homography_mask, edge_blend_radius),
            homography_mask)

        transferred = detail_transfer(canvas_blurred.crop_like(foreground.system, foreground.dims),
                                      foreground,
                                      homography_mask, detail_transfer_radius)

        print 'processed'
        return (transferred, full_mask, full_mask_blurred)

    def detail_transfer_stitch_pt_2(self, canvas_scale, detail_transfer_radius, edge_blend_radius):
        with profile():
            print 'detail_transfer_stitch_pt_2'

            canvas = self.establishing.white_balance().normalize().resize(scale=canvas_scale)
            images, full_masks, full_masks_blurred = zip(*self.detail_transfer_stitch_outputs)
            offsets = tuple(tuple(i.system.translation_origin(canvas.system).astype(int))
                            for i in images)
            homography_mask_sum = sum_arrs_with_offsets(
                (canvas.dims[1], canvas.dims[0]),
                tuple(h_m.array for h_m in full_masks),
                offsets)
            outside_mask_blurred = (
                Image((homography_mask_sum == 0).astype(float), canvas.system)
                .blur(edge_blend_radius)
            )
            self.detail_transfer_stitch_output = blend_with_offsets(
                canvas.dims,
                (canvas,) + images,
                (outside_mask_blurred,) + full_masks_blurred,
                ((0, 0),) + offsets)
            return self.detail_transfer_stitch_output

    def draw_homography_boundaries_onto(self, canvas, color=(0, 255, 0), width=10):
        for c, h in zip(self.closes, self.c_homs):
            canvas.draw_polyline(map(h, c.corners()), color=color, width=width, inplace=True)
        return canvas

    def draw_voronoi_diagram_onto(self, canvas, color=(0, 255, 0), width=10):
        for facet in self.c_facets:
            canvas.draw_polyline(clip(facet, canvas.corners()),
                                 color=color, width=width, inplace=True)
        return canvas
