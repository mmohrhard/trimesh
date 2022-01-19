"""
Ray queries using the pyembree package with the
API wrapped to match our native raytracer.
"""
import numpy as np

from copy import deepcopy

import embree
from . import parent

from .. import util
from .. import caching
from .. import intersections

from .util import contains_points
from ..constants import log_time

# the factor of geometry.scale to offset a ray from a triangle
# to reliably not hit its origin triangle
_offset_factor = 1e-4
# we want to clip our offset to a sane distance
_offset_floor = 1e-8


class RayMeshIntersector(parent.RayMeshParent):

    def __init__(self,
                 geometry,
                 scale_to_box=True):
        """
        Do ray- mesh queries.

        Parameters
        -------------
        geometry : Trimesh object
          Mesh to do ray tests on
        scale_to_box : bool
          If true, will scale mesh to approximate
          unit cube to avoid problems with extreme
          large or small meshes.
        """
        self.mesh = geometry
        self._scale_to_box = False # scale_to_box
        self._cache = caching.Cache(id_function=self.mesh.crc)

    @property
    def _scale(self):
        """
        Scaling factor for precision.
        """
        if self._scale_to_box:
            # scale vertices to approximately a cube to help with
            # numerical issues at very large/small scales
            scale = 100.0 / self.mesh.scale
        else:
            scale = 1.0
        return scale

    @caching.cache_decorator
    def _scene(self):
        """Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.
        """
        return _EmbreeWrap(vertices=self.mesh.vertices,
                           faces=self.mesh.faces,
                           scale=self._scale)

    def intersects_location(self,
                            origins,
                            directions,
                            multiple_hits=True):
        # inherits docstring from parent
        (index_tri,
         index_ray,
         locations) = self.intersects_id(
             origins=origins,
             directions=directions,
             multiple_hits=multiple_hits,
             return_locations=True)

        return locations, index_ray, index_tri

    @log_time
    def intersects_id(self,
                      origins,
                      directions,
                      multiple_hits=True,
                      max_hits=20,
                      return_locations=False):
        # inherits docstring from parent
        origins = np.asanyarray(
            deepcopy(origins),
            dtype=np.float64)
        directions = np.asanyarray(directions,
                                   dtype=np.float64)
        directions = util.unitize(directions)

        # since we are constructing all hits save them to a
        # deque then stack into (depth, len(rays)) at the end
        result_triangle = []
        result_idx = []
        result_locations = []

        # the mask for which rays are still active
        current = np.ones(len(origins), dtype=bool)

        if multiple_hits or return_locations:
            # how much to offset ray to transport to the other side of face
            distance = np.clip(_offset_factor * self._scale,
                               _offset_floor,
                               np.inf)
            offsets = directions * distance

            # grab the planes from triangles
            plane_origins = self.mesh.triangles[:, 0, :]
            plane_normals = self.mesh.face_normals

        # use a for loop rather than a while to ensure this exits
        # if a ray is offset from a triangle and then is reported
        # hitting itself this could get stuck on that one triangle
        for query_depth in range(max_hits):
            # run the pyembree query
            # if you set output=1 it will calculate distance along
            # ray, which is bizzarely slower than our calculation
            query = self._scene.run(
                origins[current],
                directions[current])

            # basically we need to reduce the rays to the ones that hit
            # something
            hit = query != -1
            # which triangle indexes were hit
            hit_triangle = query[hit]

            # eliminate rays that didn't hit anything from future queries
            current_index = np.nonzero(current)[0]
            current_index_no_hit = current_index[np.logical_not(hit)]
            current_index_hit = current_index[hit]
            current[current_index_no_hit] = False

            # append the triangle and ray index to the results
            result_triangle.append(hit_triangle)
            result_idx.append(current_index_hit)

            # if we don't need all of the hits, return the first one
            if ((not multiple_hits and
                 not return_locations) or
                    not hit.any()):
                break

            # find the location of where the ray hit the triangle plane
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=origins[current],
                line_directions=directions[current])

            if not valid.all():
                # since a plane intersection was invalid we have to go back and
                # fix some stuff, we pop the ray index and triangle index,
                # apply the valid mask then append it right back to keep our
                # indexes intact
                result_idx.append(result_idx.pop()[valid])
                result_triangle.append(result_triangle.pop()[valid])

                # update the current rays to reflect that we couldn't find a
                # new origin
                current[current_index_hit[np.logical_not(valid)]] = False

            # since we had to find the intersection point anyway we save it
            # even if we're not going to return it
            result_locations.extend(new_origins)

            if multiple_hits:
                # move the ray origin to the other side of the triangle
                origins[current] = new_origins + offsets[current]
            else:
                break

        # stack the deques into nice 1D numpy arrays
        index_tri = np.hstack(result_triangle)
        index_ray = np.hstack(result_idx)

        if return_locations:
            locations = (
                np.zeros((0, 3), float) if len(result_locations) == 0
                else np.array(result_locations))

            return index_tri, index_ray, locations
        return index_tri, index_ray

    @log_time
    def intersects_first(self,
                         origins,
                         directions):
        """
        Find the index of the first triangle a ray hits.


        Parameters
        ----------
        origins : (n, 3) float
          Origins of rays
        directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        """

        origins = np.asanyarray(deepcopy(origins))
        directions = np.asanyarray(directions)

        triangle_index = self._scene.run(origins,
                                         directions)
        return triangle_index

    def intersects_any(self,
                       origins,
                       directions):
        """
        Check if a list of rays hits the surface.


        Parameters
        -----------
        origins : (n, 3) float
          Origins of rays
        directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit the surface
        """

        first = self.intersects_first(origins=origins,
                                      directions=directions)
        hit = first != -1
        return hit

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh, behavior is undefined.

        Parameters
        ---------
        points: (n, 3) points in space

        Returns
        ---------
        contains: (n,) bool
                         Whether point is inside mesh or not
        """
        return contains_points(self, points)


class _EmbreeWrap(object):
    """
    A light wrapper for PyEmbree scene objects which
    allows queries to be scaled to help with precision
    issues, as well as selecting the correct dtypes.
    """

    def __init__(self, vertices, faces, scale):
        # TODO: figure out why scaling is not working in embree3
        # scaled = np.array(vertices, dtype=np.float64)
        # self.origin = scaled.min(axis=0)
        # self.scale = float(scale)
        # scaled = (scaled - self.origin) * self.scale
        scaled = vertices

        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        scene = device.make_scene()
        scene.set_flags(4)
        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex, # buf_type
            0, # slot
            embree.Format.Float3, # fmt
            3*np.dtype('float32').itemsize, # byte_stride
            vertices.shape[0], # item_count
        )
        vertex_buffer[:] = vertices[:].astype(np.float32)
        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index, # buf_type
            0, # slot
            embree.Format.Uint3, # fmt
            3*np.dtype('uint32').itemsize, # byte_stride,
            faces.shape[0]
        )
        index_buffer[:] = faces[:].astype(np.uint32)
        geometry.commit()
        scene.attach_geometry(geometry)
        geometry.release()
        scene.commit()

        self.scene = scene

    def run(self, origins, normals, **kwargs):
        # TODO: figure out why scaling is not working in embree3
        # scaled = (np.array(origins, dtype=np.float64) - self.origin) * self.scale
        scaled = origins

        m = origins.shape[0]

        rayhit = embree.RayHit1M(m)
        context = embree.IntersectContext()
        rayhit.org[:] = scaled.astype(np.float32)
        rayhit.dir[:] = normals.astype(np.float32)
        rayhit.tnear[:] = 0
        rayhit.tfar[:] = 1e37
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        self.scene.intersect1M(context, rayhit)

        I = rayhit.prim_id.copy().astype(np.intp)
        I[rayhit.geom_id == embree.INVALID_GEOMETRY_ID] = -1
        return I
