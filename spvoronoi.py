import cv2
import pyclipper

from spimage import ImagePoint


def clip(subj_pts, clip_pts, scale=2**16):
    system = subj_pts[0].system
    clip_coords = [p.in_system(system).coords for p in clip_pts]
    subj_coords = [p.in_system(system).coords for p in subj_pts]

    pc = pyclipper.Pyclipper()
    pc.AddPath(pyclipper.scale_to_clipper(clip_coords, scale), pyclipper.PT_CLIP, True)
    pc.AddPath(pyclipper.scale_to_clipper(subj_coords, scale), pyclipper.PT_SUBJECT, True)

    solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    return [ImagePoint(coord, system)
            for coord in pyclipper.scale_from_clipper(solution, scale)[0]]


def voronoi(points, system, dims):
    pts_here = [pt.in_system(system) for pt in points]
    subdiv = cv2.Subdiv2D((0, 0, dims[0], dims[1]))
    for pt in pts_here:
        subdiv.insert(tuple(pt.coords))

    facet_coords, _ = subdiv.getVoronoiFacetList([])

    return [[ImagePoint(coords, system) for coords in facet] for facet in facet_coords]
