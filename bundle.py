import os
import shutil
import json
import hashlib

from clint.arguments import Args
import deepzoom

from library import Library

_library = Library('_library')


def subdirs(dir):
    maybe_subdirs = (os.path.join(dir, subdir_part) for subdir_part in os.listdir(dir))
    return (maybe_subdir for maybe_subdir in maybe_subdirs if os.path.isdir(maybe_subdir))


creator = deepzoom.ImageCreator(
    tile_size=128, tile_overlap=2, tile_format="png",
    image_quality=1, resize_filter="bicubic")


def cached_deepzoom(image_path, destination):
    image_hasher = hashlib.sha1()
    with open(image_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            image_hasher.update(chunk)
    image_hash = image_hasher.hexdigest()

    def calculate(temp_dir):
        creator.create(image_path, os.path.join(temp_dir, 'image.dzi'))
    _library.get_dir(destination, calculate, image_hash, 'dzi')


def main():
    args = Args()

    input_path = os.path.realpath(args.get(0))
    output_path = os.path.realpath(args.get(1))

    image_names = ['homographies', 'masks', 'stitched']

    # Scan the input directory to get the structure
    whiteboards = {}  # whiteboards[whiteboard name] = list of stitching names
    num_stitchings = 0
    for whiteboard_path in subdirs(input_path):
        stitchings = []
        for stitching_path in subdirs(whiteboard_path):
            files = [os.path.join(stitching_path, part + '.png') for part in image_names]
            if all(os.path.exists(file) for file in files):
                stitchings.append(os.path.basename(stitching_path))
                num_stitchings += 1
        if stitchings:
            whiteboards[os.path.basename(whiteboard_path)] = stitchings

    # Remake the output directory
    shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Write the structure
    with open(os.path.join(output_path, 'whiteboards.json'), 'w') as f:
        json.dump(whiteboards, f)

    # Write the deepzooms
    num_deepzooms_written = 0
    for whiteboard_name, stitchings in whiteboards.iteritems():
        input_whiteboard_path = os.path.join(input_path, whiteboard_name)
        output_whiteboard_path = os.path.join(output_path, whiteboard_name)
        os.mkdir(output_whiteboard_path)
        for stitching_name in stitchings:
            input_stitching_path = os.path.join(input_whiteboard_path, stitching_name)
            output_stitching_path = os.path.join(output_whiteboard_path, stitching_name)
            os.mkdir(output_stitching_path)
            for image_name in image_names:
                print (num_deepzooms_written + 1), '/', len(image_names) * num_stitchings
                cached_deepzoom(
                    os.path.join(input_stitching_path, image_name + '.png'),
                    os.path.join(output_stitching_path, image_name)
                )
                num_deepzooms_written += 1


if __name__ == "__main__":
    main()
