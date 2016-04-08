import stitching
import spimage

establishing = spimage.Image.from_file('establishing.jpg')
closes = map(spimage.Image.from_file, [
    '2016-03-14 17.46.19.jpg', '2016-03-14 17.46.43.jpg', '2016-03-14 17.47.18.jpg',
    '2016-03-14 17.47.22.jpg', '2016-03-14 17.47.26.jpg', '2016-03-14 17.47.30.jpg',
    '2016-03-14 17.47.34.jpg', '2016-03-14 17.47.38.jpg', '2016-03-14 17.47.42.jpg',
    '2016-03-14 17.47.50.jpg', '2016-03-14 17.47.53.jpg', '2016-03-14 17.47.57.jpg',
    '2016-03-14 17.48.04.jpg', '2016-03-14 17.48.08.jpg', '2016-03-14 17.48.11.jpg',
    '2016-03-14 17.48.15.jpg', '2016-03-14 17.48.19.jpg'
])

job = stitching.StitchingJob(establishing, closes)

job.find_homographies(downsample_scale=0.5)
job.find_voronoi()

detail_transfer_stitch = job.detail_transfer_stitch(canvas_scale=3, detail_transfer_radius=33,
    edge_blend_radius=5, blend_it=True)

print 'saving'
detail_transfer_stitch.save('objmod.png')
print 'done'
