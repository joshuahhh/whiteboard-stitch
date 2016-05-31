import os
import glob
import sys

from clint.arguments import Args
import yaml

import stitching
import spimage

if __name__ == "__main__":
    args = Args()

    config_path_rel = args.get(0)

    config = yaml.load(open(config_path_rel))

    c_files = config['files']
    c_feature_detection = config['feature_detection']
    c_stitching = config['stitching']

    config_dir = os.path.dirname(os.path.realpath(config_path_rel))

    establishing = spimage.Image.from_file(os.path.join(config_dir, c_files['establishing']))
    # TODO: allow arrays of images rather than just globs
    closes = map(spimage.Image.from_file, glob.glob(os.path.join(config_dir, c_files['closes'])))

    output_dir = os.path.join(
        config_dir,
        os.path.basename(config_path_rel).replace('.', '_') + '_out')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    job = stitching.StitchingJob(establishing, closes)

    # Make sure to stay in the root directory for finding homographies, so
    # features can be cached!
    job.find_homographies(**c_feature_detection)

    canvas = job.establishing.copy()
    job.draw_homography_boundaries_onto(canvas)
    canvas.save(os.path.join(output_dir, 'homographies.png'))

    job.calculate_areas()
    if 'canvas_scale' in c_stitching:
        job.canvas_scale = c_stitching['canvas_scale']
        del c_stitching['canvas_scale']
    else:
        job.calculate_canvas_scale()

    partition_method = config.get('partition_method', 'voronoi')
    if partition_method == 'voronoi':
        job.generate_masks_voronoi()
    elif partition_method == 'stacked':
        job.generate_masks_stacked()
    else:
        raise Exception('Unknown partition_method')

    canvas = job.establishing.copy()  # .resize(scale=job.canvas_scale)
    job.draw_mask_boundaries_onto(canvas)
    job.draw_masks_onto(canvas)
    canvas.save(os.path.join(output_dir, 'masks.png'))

    if '-n' in args.flags:
        # Don't actually stitch it.
        sys.exit(0)

    if 'detail_transfer_sigma_color' in c_stitching:
        # For the closure:
        radius = c_stitching['detail_transfer_radius']
        sigma_color = c_stitching['detail_transfer_sigma_color']
        sigma_space = c_stitching['detail_transfer_sigma_space']
        c_stitching['detail_transfer_blur_op'] = (
            lambda im: im.bilateral_blur(radius, sigma_color, sigma_space))
        del c_stitching['detail_transfer_radius']
        del c_stitching['detail_transfer_sigma_color']
        del c_stitching['detail_transfer_sigma_space']
    else:
        # For the closure:
        radius = c_stitching['detail_transfer_radius']
        c_stitching['detail_transfer_blur_op'] = lambda im: im.blur(radius)
        del c_stitching['detail_transfer_radius']

    job.detail_transfer_stitch_pt_1(**c_stitching)
    job.detail_transfer_stitch_pt_2(**c_stitching)

    print 'saving'
    job.detail_transfer_stitch_output.save(os.path.join(output_dir, 'stitched.png'))
    print 'done'
