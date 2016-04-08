import stitching
import spimage

establishing = spimage.Image.from_file('conc/2016-04-05 11.07.33.jpg')
closes = map(spimage.Image.from_file, [
    'conc/2016-04-05 11.07.44.jpg',
    'conc/2016-04-05 11.07.50.jpg',
    'conc/2016-04-05 11.07.56.jpg',
    'conc/2016-04-05 11.08.01.jpg'
])

job = stitching.StitchingJob(establishing, closes)

job.find_homographies(downsample_scale=0.5)
job.find_voronoi()

detail_transfer_stitch = job.detail_transfer_stitch(canvas_scale=2, detail_transfer_radius=33,
    edge_blend_radius=5)
detail_transfer_stitch.show()
