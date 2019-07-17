#!/usr/bin/env python
import numpy as np
from scipy import interpolate
import os, sys
import struct
import boundary as boundary
import pickle

"""Module for handling meshes with i/o support for ply and stl filetypes. 
Also allows relaxation of spring network defined by the mesh.

written NPMitchell and DKleckner
npmitchell@kitp.ucsb.edu
"""

import sys
if sys.version_info[0] < 3:
    py2 = True
else:
    py2 = False

def struct_read(f, fs):
    s = struct.Struct(fs)
    return s.unpack(f.read(s.size))


class Mesh(object):
    def __init__(self, points=np.zeros((0, 3)), triangles=np.zeros((0, 3))):
        """To load a mesh from PLY or STL, use filename as the value for the kwarg 'points'
        """
        self.points = None
        self.triangles = None
        self.normals = None
        self.vertex_normals = None

        if type(points) == str:
            ext = os.path.splitext(points)[1].lower()

            if ext == '.ply':
                print('mesh: extension is ply...')
                f = open(points, 'rt')
                # print('f.readline() = ', f.readline()
                if 'ply' not in f.readline():
                    print('mesh: Reading mesh stored as ply...')
                    if 'format ascii' not in f.readline():
                        raise ValueError("File is not ascii formatted PLY!  I don't know how to read it.")

                line = ''
                while 'end_header' not in line:
                    line = f.readline()
                    if line.startswith('element'):
                        parts = line.split()
                        if parts[1] == 'vertex':
                            self.num_v = int(parts[2])
                            print('mesh: determined number of vertices = ' + str(self.num_v))
                        elif parts[1] == 'face':
                            self.num_f = int(parts[2])
                            print('mesh: determined number of faces = ' + str(self.num_f))

                if not hasattr(self, 'num_v'):
                    raise ValueError("Couldn't find number of vertices in PLY file.")

                if not hasattr(self, 'num_f'):
                    raise ValueError("Couldn't find number of faces in PLY file.")

                self.points = np.zeros((self.num_v, 3))
                for ii in range(self.num_v):
                    line = f.readline()
                    try:
                        xx = list(map(float, line.split()))
                    except:
                        xx = []

                    if not len(xx) in [3, 6]:
                        raise ValueError("Entry in PLY file (%s) doesn't look like a vertex!" % line)

                    # Prepare the vertex_normals array, if they are to be read from the file
                    if ii == 0 and len(xx) > 3:
                        self.vertex_normals = np.zeros_like(self.points)

                    if len(xx) == 3:
                        self.points[ii] = xx
                    elif len(xx) == 6:
                        # The line has both a vertex and the normal at that vertex
                        self.points[ii] = xx[0:3]
                        self.vertex_normals[ii] = xx[3:]

                self.triangles = np.zeros((self.num_f, 3), dtype='u4')
                for i in range(self.num_f):
                    line = f.readline()
                    try:
                        X = list(map(int, line.split()))
                    except:
                        X = []
                    if len(X) != 4 and X[0] != 3:
                        raise ValueError(
                            "Entry in PLY file (%s) doesn't look like a triangle!\n"
                            "(This script does not accept quads, etc.)" % line)
                    else:
                        self.triangles[i] = X[1:]

            elif ext == '.stl':
                f = open(points, 'rb')

                if f.read(5).lower() == 'solid':
                    raise ValueError("ASCII STL reading not implemented!")

                f.seek(80)
                num_triangles = struct_read(f, 'I')[0]
                print('num_triangles = ', num_triangles)

                self.points = np.zeros((num_triangles * 3, 3), dtype='d')
                self.triangles = np.arange(num_triangles * 3, dtype='u4').reshape((num_triangles, 3))
                self.vertex_normals = np.zeros_like(self.points)

                j = 0
                for i in range(num_triangles):
                    nx, ny, nz, ax, ay, az, bx, by, bz, cx, cy, cz, att = struct_read(f, '12fH')

                    for k, p in enumerate([(ax, ay, az), (bx, by, bz), (cx, cy, cz)]):
                        jj = j + k

                        self.points[jj] = p
                        self.vertex_normals[jj] = (nx, ny, nz)

                    j += 3

            elif ext == '.off':
                self.points = []
                self.triangles = []
                self.load_off(points)
            else:
                raise ValueError("Only STL/PLY files supported for loading.")
        else:
            self.points = np.array(points)
            self.triangles = np.array(triangles, dtype='u4')

    def inverted(self):
        return Mesh(self.points.copy(), self.triangles[:, ::-1].copy())

    def translate(self, offset):
        return Mesh(self.points + offset, self.triangles.copy())

    def scale(self, s):
        s = np.asarray(s)
        if not s.shape:
            s = s * ones(3)
        return Mesh(self.points * s, self.triangles.copy())

    def draw_triangles(self, draw_func, with_z=False, close=True, *args, **kwargs):
        for t in self.triangles:
            if close:
                t = np.hstack((t, t[0:1]))
            if with_z:
                x, y, z = self.points[t, :3].T
                draw_func(x, y, z, *args, **kwargs)
            else:
                x, y = self.points[t, :2].T
                draw_func(x, y, *args, **kwargs)

    def copy(self):
        return Mesh(self.points.copy(), self.triangles.copy())

    def volume(self):
        px, py, pz = self.tps(0).T
        qx, qy, qz = self.tps(1).T
        rx, ry, rz = self.tps(2).T

        return (px * qy * rz + py * qz * rx + pz * qx * ry - px * qz * ry - py * qx * rz - pz * qy * rx).sum() / 6.

    def is_closed(self, tol=1E-12):
        x, y, z = self.points.T
        m2 = self.copy()
        m2.points += 2 * np.array((max(x) - min(x), max(y) - min(y), max(z) - min(z)))
        v1 = self.volume()
        v2 = m2.volume()
        return abs((v1 - v2) / v1) < tol

    def __add__(self, other):
        if hasattr(other, 'points') and hasattr(other, 'triangles'):
            return Mesh(
                points=np.vstack((self.points, other.points)),
                triangles=np.vstack((self.triangles, other.triangles + len(self.points)))
            )

        else:
            raise TypeError('Can only add a Mesh to another Mesh')

    def tps(self, n):
        return self.points[self.triangles[:, n]]

    def make_normals(self, normalize=True):
        n = np.cross(self.tps(2) - self.tps(0), self.tps(1) - self.tps(0))
        if normalize:
            n = norm(n)

        self.normals = n
        return n

    def make_vertex_normals(self):
        """Define the normals on each vertex"""
        if not hasattr(self, 'normals'):
            self.make_normals()

        self.vertex_normals = np.zeros_like(self.points)
        for i, n in enumerate(self.normals):
            for j in self.triangles[i]:
                self.vertex_normals[j] += n

        self.vertex_normals = norm(self.vertex_normals)
        return self.vertex_normals

    def force_z_normal(self, direction=1):
        """

        Parameters
        ----------
        direction : int (1 or -1) or signed float
            direction to enforce normals to be pointing

        Returns
        -------

        """
        if not hasattr(self, 'normals'): self.make_normals()

        inverted = np.where(np.sign(self.normals[:, 2]) != np.sign(direction))[0]
        self.triangles[inverted] = self.triangles[inverted, ::-1]
        self.normals[inverted] *= -1

    def load_off(self, fn):
        """Reads vertices and faces from an off file.


        Parameters
        ----------
        file : str
            path to file to read

        Returns
        -------
        vertices and faces as lists of tuples
            [(float)], [(int)]
        """

        assert os.path.exists(fn)

        with open(fn, 'r') as fp:
            lines = fp.readlines()
            lines = [line.strip() for line in lines]

            assert lines[0] == 'OFF'

            # Find the first non-empty line with no comment
            offset, found = 1, False
            while not found:
                if len(lines[offset]) > 0:
                    if not lines[offset][0] in ['#', ' ']:
                        found = True
                    else:
                        offset += 1
                else:
                    offset += 1

            # Grab number of vertices (points) and faces (triangles)
            # The number of edges is ignored
            parts = lines[offset].split(' ')
            assert len(parts) == 3
            num_vertices = int(parts[0])
            assert num_vertices > 0
            num_faces = int(parts[1])
            assert num_faces > 0

            # Advance to the next line
            offset += 1

            # Is the next line blank?
            found = False
            while not found:
                if len(lines[offset]) > 0:
                    if not lines[offset][0] in ['#', ' ']:
                        found = True
                    else:
                        offset += 1
                else:
                    offset += 1

            # Load all vertices (points) and faces (triangles)
            vertices = []
            for i in range(num_vertices):
                vertex = lines[offset + i].split(' ')
                vertex = [float(point) for point in vertex]
                assert len(vertex) == 3

                vertices.append(vertex)

            faces = []
            for i in range(num_faces):
                if len(lines[offset + num_vertices + i].split('  ')) > 1:
                    face = lines[offset + num_vertices + i].split('  ')[-1].split(' ')
                else:
                    face = lines[offset + num_vertices + i].split(' ')
                    if len(face) == 4:
                        face = face[1:4]

                face = [int(index) for index in face]

                for index in face:
                    assert index >= 0 and index < num_vertices

                assert len(face) > 1

                faces.append(face)

            self.points = vertices
            self.triangles = faces
            return vertices, faces

    def save(self, fn, ext=None):
        if ext is None:
            ext = os.path.splitext(fn)[-1][1:]

        ext = ext.lower()
        if not ext in ['ply', 'stl', 'xml', 'obj', 'ply']:
            print('Extension not recognized: ext = ' + ext + '\nUsing last three characters from fn...')
            ext = fn[-3:]

        if ext == 'ply':
            self.save_ply(fn)
        elif ext == 'stl':
            self.save_stl(fn)
        elif ext == 'xml':
            self.save_xml(fn)
        elif ext == 'obj':
            self.save_obj(fn)
        elif ext == 'off':
            self.save_off(fn)
        else:
            raise ValueError("Extension should be 'stl' 'ply' 'xml' 'obj' 'off' for outputting mesh")

    def save_stl(self, fn, header=None):
        output = open(fn, 'wb')
        if header is None:
            header = '\x00\x00This is an STL file. (http://en.wikipedia.org/wiki/STL_(file_format))'

        e = '<'

        output.write(header + ' ' * (80 - len(header)))
        output.write(struct.pack(e + 'L', len(self.triangles)))

        self.make_normals()

        for t, n in zip(self.triangles, self.normals):
            output.write(struct.pack(e + 'fff', *n))
            for p in t:
                x = self.points[p]
                output.write(struct.pack(e + 'fff', *x))
            output.write(struct.pack(e + 'H', 0))

        output.close()

    def save_ply(self, fn, save_normals=False):
        """Save the current Mesh instance as a PLY file.

        Parameters
        ----------
        fn : str
            the path to the output file
        save_normals : bool
            whether to include normal vectors at each vertex

        Returns
        -------

        """
        output = open(fn, 'wt')

        if save_normals:
            output.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property float nx
            property float ny
            property float nz
            element face %d
            property list uchar int vertex_indices
            end_header
            ''' % (len(self.points), len(self.triangles)))

            for (pp, nn) in zip(self.points, self.vertex_normals):
                output.write('%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n' % tuple(np.hstack((pp, nn))))
        else:
            output.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            element face %d
            property list uchar int vertex_indices
            end_header
            ''' % (len(self.points), len(self.triangles)))

            for p in self.points:
                output.write('%10.5f %10.5f %10.5f\n' % tuple(p))

        for t in self.triangles:
            output.write('3 %5d %5d %5d\n' % tuple(t))

        output.close()

    def save_xml(self, fname):
        """Create a dolfin (FEniCS) mesh from Mesh mesh by saving it to a FEniCS-compatible XML file.
        The input mesh can be 2D or 3D.

        Parameters
        ----------
        mesh : instance of Mesh class
            a mesh created using ilpm.mesh
        fname : string
            the complete filename path for outputting as xml

        Returns
        ----------
        out : either string or fenics mesh
            If dolfin is available in your current python environment, returns the input mesh as dolfin mesh.
            Otherwise, saves the mesh file but returns a string statement that could not import dolfin from current environment.
        """
        pts = self.points
        tri = self.triangles

        if np.shape(pts)[1] == 2:
            is2d = True
        else:
            is2d = False

        ################
        # Write header #
        ################
        print('Writing header...')
        with open(fname, 'w') as myfile:
            myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        with open(fname, 'a') as myfile:
            myfile.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
            myfile.write('  <mesh celltype="triangle" dim="2">\n')
            myfile.write('    <vertices size="' + str(np.shape(pts)[0]) + '">\n')

        # write vertices (positions)
        print('Writing vertices...')
        with open(fname, 'a') as myfile:
            for (pt, i) in zip(pts, range(len(pts))):
                if is2d:
                    myfile.write('      <vertex index="' + str(i) + '" x="' + '{0:.9E}'.format(pt[0]) +
                                 '" y="' + '{0:.9E}'.format(pt[1]) + '"/>\n')
                else:
                    myfile.write('      <vertex index="' + str(i) + '" x="' + '{0:.9E}'.format(pt[0]) +
                                 '" y="' + '{0:.9E}'.format(pt[1]) + '" z="' + '{0:.9E}'.format(pt[2]) + '"/>\n')

            myfile.write('    </vertices>\n')

        # write the bonds
        print('Writing triangular elements...')
        with open(fname, 'a') as myfile:
            myfile.write('    <cells size="' + str(len(tri)) + '">\n')
            for i in range(len(tri)):
                myfile.write('      <triangle index="' + str(i) + '" v0="' + str(tri[i, 0]) +
                             '" v1="' + str(tri[i, 1]) + '" v2="' + str(tri[i, 2]) + '"/>\n')

            myfile.write('    </cells>\n')

        with open(fname, 'a') as myfile:
            myfile.write('  </mesh>\n')

        with open(fname, 'a') as myfile:
            myfile.write('</dolfin>')

        print('done! Wrote mesh to: ', fname)
        try:
            import dolfin as dolf
            print('Attempting to load meshfile in dolfin = ', fname)
            dolf.Mesh(fname)
            return self
        except ImportError:
            print('Wrote xml file but could not import dolfin in current python environment.')
            return self
        except:
            print('Wrote xml file but could not load it in dolfin.')
            return self

    def save_obj(self, fn, nm_fn=None, mtl=False):
        """Write a mesh to an OBJ file. Adapted from MATLAB code by Gabriel Peyre


        Parameters
        ----------
        fn : str
            The full path of the OBJ file to save
        nm_fn : str
            The path of the material to save/load? using mtllib
        mtl : bool
            Whether to include information about the material properties? Not really sure -- perhaps refer to Gabriel
            Peyre's code to understand what exactly this does

        Returns
        -------

        """
        vertex = self.points
        face = self.triangles
        if np.shape(vertex)[1] != 3:
            vertex = np.transpose(vertex)

        if np.shape(vertex)[1] != 3:
            raise RuntimeError('vertex does not have the correct format.')

        if face is not None:
            if np.shape(face)[1] != 3:
                face = np.transpose(face)

            if np.shape(face)[1] != 3:
                raise RuntimeError('face does not have the correct format.')

        with open(fn, 'wt') as output:
            object_name = fn[:-4]
            output.write('# Created via mesh.Mesh.save_obj() (c) 2019 Noah Mitchell\n# \n')
            if nm_fn is not None:
                output.write('mtllib ./' + object_name + '.mtl\n')

            if mtl:
                object_name = 'curobj'
                output.write('g\n# object ' + object_name + ' to come\n')

            # vertex positions
            output.write('# %d vertex\n' % np.shape(vertex)[0])
            for pt in vertex:
                output.write('v %10.5f %10.5f %10.5f\n' % tuple(pt))

            # vertex texture
            if nm_fn is not None:
                nvert = np.shape(vertex)[0]
                object_texture = np.zeros((nvert, 2))
                m = np.ceil(np.sqrt(nvert))
                if m ** 2 != nvert:
                    raise RuntimeError('To use normal map the number of vertex must be a square.');

                xa = np.arange(0, 1, 1. / (m - 1))
                [Yy, Xx] = np.meshgrid(xa, xa)
                object_texture[:, 1] = Yy.ravel()
                object_texture[:, 2] = np.fliplr(Xx.ravel())
                for ot in object_texture:
                    output.write('vt %f %f\n', ot)

            # use mtl
            if mtl:
                output.write('g ' + object_name + '_export\n')
                mtl_bump_name = 'bump_map'
                output.write('usemtl ' + mtl_bump_name + '\n')

            # write faces to file
            if len(face) > 0:
                output.write('# %d faces\n' % np.shape(face)[0])
                if mtl:
                    # Include textrure positions to be the same as vertex positions by using slash notation.
                    # Note that we could also include normal directions using one further slash.
                    face_texcorrd = np.dstack((face[:, 0], face[:, 0], face[:, 1], face[:, 1],
                                               face[:, 2], face[:, 2]))[0]
                    for ftc in face_texcorrd:
                        output.write('f %d/%d %d/%d %d/%d\n' % tuple(ftc))
                else:
                    face_texcorrd = np.dstack((face[:, 0], face[:, 1], face[:, 2]))[0]
                    for ftc in face_texcorrd:
                        output.write('f %d %d %d\n' % tuple(ftc))

    def save_off(self, fn, comment_header=None):
        """Saves the mesh to an OFF file.

        Parameters
        ----------
        fn : str
            path to file to read
        comment_header : str or None (optional)
            Comment to place at top of file (after 'OFF')

        Returns
        -------
        vertices and faces as lists of tuples
            [(float)], [(int)]
        """
        with open(fn, 'w') as output:
            # First line says 'OFF', second line gives #vertices, #faces, #edges or 0
            output.write('OFF\n')
            if comment_header is not None:
                output.write(comment_header)
                # Make sure comment ends in new line
                if '\n' not in comment_header[-2:]:
                    output.write('\n')

            output.write('\n')
            output.write('%d %d %d\n' % (len(self.points), len(self.triangles), 0))

            num_vertices = np.shape(self.points)[0]
            num_faces = np.shape(self.triangles)[0]
            if num_vertices == 0:
                raise RuntimeError('No points in mesh to save in OFF file')
            if num_faces == 0:
                raise RuntimeError('No triangles in mesh to save in OFF file')

            for pt in self.points:
                output.write('%.9f %.9f %.9f\n' % tuple(pt))
            for tri in self.triangles:
                output.write('3  %d %d %d \n' % tuple(tri))

    def relax_z(self, fixed=None, steps=5):
        """Minimize energy of spring network made of mesh, to make smooth shapes

        Parameters
        ----------
        fixed
        steps

        Returns
        -------

        """
        oz = self.points[:, 2]
        N = len(self.points)
        K = dict()  # Stiffness matrix

        dist = lambda p1, p2: np.sqrt(sum((self.points[p1, :2] - self.points[p2, :2]) ** 2))

        for t in self.triangles:
            # Triangle side lengths
            a = dist(t[1], t[2])
            b = dist(t[2], t[0])
            c = dist(t[0], t[1])
            s = (a + b + c) / 2
            # Heron's formula
            A = np.sqrt(s * (s - a) * (s - b) * (s - c))

            p1, p2 = t[1], t[2]
            if p1 > p2: p1, p2 = p2, p1
            pair = (p1, p2)
            K[pair] = K.get(pair, 0.) + (-a ** 2 + b ** 2 + c ** 2) / A

            p1, p2 = t[2], t[0]
            if p1 > p2: p1, p2 = p2, p1
            pair = (p1, p2)
            K[pair] = K.get(pair, 0.) + (a ** 2 - b ** 2 + c ** 2) / A

            p1, p2 = t[0], t[1]
            if p1 > p2: p1, p2 = p2, p1
            pair = (p1, p2)
            K[pair] = K.get(pair, 0.) + (a ** 2 + b ** 2 - c ** 2) / A

        # nc = histogram(nl.flatten(), bins=range(-1, N+2))[0][1:-1]
        # nc[np.where(nc == 0)] = 1
        tK = np.zeros(N)
        for (p1, p2), W in K.iteritems():
            tK[p1] += W
            tK[p2] += W

        tK[np.where(tK == 0)] = 1

        for n in range(steps):
            z = np.zeros(N)

            for (p1, p2), W in K.iteritems():
                z[p1] += W * oz[p2]
                z[p2] += W * oz[p1]

            z /= tK

            if fixed is not None:
                z[fixed] = self.points[fixed, 2]

            oz = z

        self.points[:, 2] = z

    def rot_x(self, angle):
        return Mesh(rot_x(self.points, angle), self.triangles)

    def rot_y(self, angle):
        return Mesh(rot_y(self.points, angle), self.triangles)

    def rot_z(self, angle):
        return Mesh(rot_z(self.points, angle), self.triangles)

    def merge_points(self, tol=1E-10, verbose=False):
        new = np.zeros((len(self.points), 3))
        p_map = np.zeros(len(self.points), dtype='i')

        if verbose:
            print('Merging %d points...' % len(self.points))

        j = 0
        for i, p in enumerate(self.points):
            if j == 0:
                new[j] = p
                p_map[i] = j
                j += 1
            else:
                dist = m1(new[:j] - p)
                j_min = np.argmin(dist)
                min_dist = dist[j_min]
                if min_dist < tol:
                    p_map[i] = j_min
                else:
                    new[j] = p
                    p_map[i] = j
                    j += 1

        print('   Done.  Eliminated %d redundant points.' % (len(self.points) - j))
        self.points = new[:j]
        self.triangles = p_map[self.triangles]

    def project(self, X, x, y, z=None):
        if z is None:
            z = np.cross(x, y)

        qq = np.zeros_like(self.points)
        for i, a in enumerate((x, y, z)):
            qq += self.points[:, i:i + 1] * a

        qq += X

        return Mesh(qq, self.triangles.copy())


def closed_path_interp(X, threshold=2., interp_func=interpolate.interp1d, interp_args=(), interp_kwargs={'axis': 0}):
    d = threshold * mag(X[1] - X[0])
    lead = True

    for i, x in enumerate(X[2:]):
        if mag(x - X[0]) < d:
            if lead is False:
                print("Found path closure at %d points" % i)
                break
        else:
            lead = False
    else:
        print("Warning: didn't find path closure, joining first and last point...")

    X = np.vstack((X[:i], X[0:1]))

    l = np.zeros(len(X))
    for i in range(1, len(X)):
        l[i] = mag(X[i] - X[i - 1]) + l[i - 1]

    total_l = l[-1]
    interp = interp_func(l, X, *interp_args, **interp_kwargs)
    return total_l, lambda x: interp(x % total_l)


def Gram_Schmidt(*vecs):
    ortho = []

    for v in vecs:
        v = np.array(v)
        for b in ortho:
            v -= b * np.dot(v, b)
        ortho.append(norm(v))

    return ortho


def make_cap(p, direction=1, offset=0):
    x, y = Gram_Schmidt(p[1] - p[0], p[2] - p[0])

    p2 = np.array([np.dot(p, x), np.dot(p, y), np.zeros(len(p))]).T
    b = boundary.Boundary(p2)

    m = b.mesh_cap(direction=direction, ignore_wind=True)

    return m.triangles + offset


def make_tube(points, nc, cap=False, invert_caps=False):
    N = int(len(points) // nc)
    if nc * N != len(points):
        raise ValueError('Number of points in a tube must be a multiple of the number around the circumference!')
    tris = []

    for i in range(N - (1 if cap else 0)):
        i0 = i * nc
        i1 = ((i + 1) % N) * nc

        for j0 in range(nc):
            j1 = (j0 + 1) % nc
            tris += [(i0 + j0, i0 + j1, i1 + j0), (i1 + j0, i0 + j1, i1 + j1)]

    if cap:
        direction = -1 if invert_caps else 1

        tris += list(make_cap(points[:nc], direction=-1 * direction))
        tris += list(make_cap(points[-nc:], direction=direction, offset=len(points) - nc))

    return Mesh(points, tris)


def arglocalmin(x):
    return list(np.where((x < shift(x)) * (x <= shift(x, -1)))[0])


def cone(x0, x1, r0, r1=None, a=None, points=10):
    N = norm(np.array(x1) - x0)
    if a is None: a = np.array((1, 0, 0)) if abs(N[0]) < 0.9 else np.array((0, 1, 0))
    A = norm(a - proj(a, N))
    B = np.cross(N, A)
    phi = np.arange(points) * 2 * np.pi / points
    phi.shape += (1,)

    C = np.cos(phi) * A + np.sin(phi) * B

    if r1 is None:
        p = np.vstack((np.array((x1)), x0 + C * r0))
        t = [(0, n + 1, (n + 1) % points + 1) for n in range(points)] + \
            [(1, 1 + (n + 1) % points, n + 1) for n in range(1, points - 1)]

    else:
        p = np.vstack((x0 + C * r0, x1 + C * r1))
        t = [(n, (n + 1) % points + points, n + points) for n in range(points)] + \
            [(n, (n + 1) % points, (n + 1) % points + points) for n in range(points)] + \
            [(0, (n + 1) % points, n) for n in range(1, points - 1)] + \
            [(0 + points, n + points, (n + 1) % points + points) for n in range(1, points - 1)]

    return Mesh(p, t)


def column(x0, x1, radius_function=lambda x: 1 + 2 * (x - x ** 2), rp=20, lp=20):
    N = norm(np.array(x1) - x0)
    if a is None: a = np.array((1, 0, 0)) if abs(N[0]) < 0.9 else np.array((0, 1, 0))
    A = norm(a - proj(a, N))
    B = np.cross(N, A)
    phi = np.arange(points) * 2 * np.pi / points
    phi.shape += (1,)

    C = np.cos(phi) * A + np.sin(phi) * B

    if r1 is None:
        p = np.vstack((np.array((x1)), x0 + C * r0))
        t = [(0, n + 1, (n + 1) % points + 1) for n in range(points)] + \
            [(1, 1 + (n + 1) % points, n + 1) for n in range(1, points - 1)]

    else:
        p = vstack((x0 + C * r0, x1 + C * r1))
        t = [(n, (n + 1) % points + points, n + points) for n in range(points)] + \
            [(n, (n + 1) % points, (n + 1) % points + points) for n in range(points)] + \
            [(0, (n + 1) % points, n) for n in range(1, points - 1)] + \
            [(0 + points, n + points, (n + 1) % points + points) for n in range(1, points - 1)]

    return Mesh(p, t)


def arrow(x0, x2, points=10):
    x1 = (np.array(x0) + x2) / 2.
    l = mag(np.array(x0) - x2)
    return cone(x0, x1, l / 10., l / 10., points=points) + \
           cone(x1, x2, l / 4., points=points)


def circle(c=np.zeros(3), r=1, numpts=100):
    theta = np.arange(numpts, dtype='d') / numpts * 2 * np.pi
    p = np.zeros((numpts, len(c)))  # c might be 3D!
    p[:] = c
    p[:, 0] += r * np.cos(theta)
    p[:, 1] += r * np.sin(theta)
    return p


def rot_x(x, a):
    x = np.array(x)
    rx = x.copy()
    rx[..., 1] = np.cos(a) * x[..., 1] - np.sin(a) * x[..., 2]
    rx[..., 2] = np.cos(a) * x[..., 2] + np.sin(a) * x[..., 1]

    return rx


def rot_y(x, a):
    x = np.array(x)
    rx = x.copy()
    rx[..., 2] = np.cos(a) * x[..., 2] - np.sin(a) * x[..., 0]
    rx[..., 0] = np.cos(a) * x[..., 0] + np.sin(a) * x[..., 2]

    return rx


def rot_z(x, a):
    x = np.array(x)
    rx = x.copy()
    rx[..., 0] = np.cos(a) * x[..., 0] - np.sin(a) * x[..., 1]
    rx[..., 1] = np.cos(a) * x[..., 1] + np.sin(a) * x[..., 0]

    return rx


def shift(a, n=1):
    return a[(np.arange(len(a)) + n) % len(a)]


def mag(x, axis=None):
    x = np.asarray(x)
    if len(x.shape) == 1:
        return np.sqrt((x ** 2).sum())
    else:
        if axis is None: axis = len(x.shape) - 1
        m = np.sqrt((x ** 2).sum(axis))
        ns = list(x.shape)
        ns[axis] = 1
        return m.reshape(ns)


def m1(x):
    return np.sqrt((x ** 2).sum(1))


def D(x):
    x = np.array(x)
    return 0.5 * (shift(x, +1) - shift(x, -1))


def norm(x):
    return x / mag(x)


def path_frame(x, curvature=False):
    dr = D(x)
    ds = mag(dr)
    T = dr / ds
    N = norm(D(T))
    B = np.cross(T, N)

    if curvature:
        return T, N, B, mag(D(T) / ds)
    else:
        return T, N, B


def N_2D(x):
    return norm(np.cross((0, 0, 1), norm(D(x))))


def proj(a, b):
    b = norm(b)
    return a.dot(b) * b


def bezier(p0, p1, p2, p3, numpts=10, include_ends=False):
    if include_ends:
        t = (np.arange(numpts + 2, dtype='d')) / (numpts + 1)
    else:
        t = (np.arange(numpts, dtype='d') + 1) / (numpts + 1)
    t.shape = t.shape + (1,)

    pt = lambda n: (1 - t) ** (3 - n) * t ** n

    return pt(0) * p0 + 3 * (pt(1) * p1 + pt(2) * p2) + pt(3) * p3


rot_axes = {'x': rot_x, 'y': rot_y, 'z': rot_z}


def surface_of_revolution(path, start=None, end=None, numpts=100, axis='z'):
    if start is None:
        theta = np.arange(numpts, dtype='d') / numpts * 2 * np.pi
        cap = False
    else:
        theta = np.linspace(start, end, np.ceil(abs(end - start) / (2 * np.pi) * numpts) + 1)
        cap = True

    p = []
    rot = rot_axes[axis.lower()]
    for t in theta:
        p += list(rot(path, t))

    return make_tube(p, len(path), cap=cap)


def path_connection(p1, s1, p2, s2, numpts):
    d1 = p1[-1] - p1[-2]
    d1 /= np.sqrt(sum(d1 ** 2))

    d2 = p2[0] - p2[1]
    d2 /= np.sqrt(sum(d2 ** 2))

    return bezier(p1[-1], p1[-1] + s1 * d1, p2[0] + s2 * d2, p2[0], numpts, False)


def arc(c, r, a1, a2, numpts=20, max_error=None):
    if max_error is not None:
        numpts = max(abs(a1 - a2) / 2. * np.sqrt(r / max_error), 3)

    phi = np.linspace(a1, a2, numpts)

    x = np.zeros((np, len(c)))
    x[:] = c
    x[..., 0] += r * np.cos(phi)
    x[..., 1] += r * np.sin(phi)

    return x


def join_meshes(m1, m2, boundaries):
    numpts = len(m1.points)
    m = m1 + m2

    edge_tris = []
    n0 = 0
    for p in boundaries.paths:
        pip = len(p)

        for j in range(pip):
            j1 = n0 + j
            j2 = n0 + (j + 1) % pip
            j3 = j1 + numpts
            j4 = j2 + numpts

            edge_tris += [(j3, j2, j1), (j2, j3, j4)]

        n0 += pip

    m.triangles = np.vstack((m.triangles, edge_tris))

    return m


def circular_array(x, delta, n, rot=rot_z, offset=0):
    return np.vstack([rot(x, i * delta + offset) for i in np.arange(n)])


def oriented_tube(path, outline):
    z = np.array((0, 0, 1), dtype='d')
    T, N, B = path_frame(path)
    NP = norm(N * (1 - z))
    x = outline[:, 0:1]
    y = outline[:, 1:2]

    points = []
    for p, n in zip(path, NP):
        points += list(p + n * x + z * y)

    return make_tube(points, len(outline))


def oriented_offset(path, offset):
    z = np.array((0, 0, 1), dtype='d')
    T, N, B = path_frame(path)
    NP = norm(N * (1 - z))

    x, y = offset
    return path + NP * x + z * y


def arglocalmin(x):
    return list(np.where((x < shift(x)) * (x <= shift(x, -1)))[0])


def arglocalmax(x):
    return list(np.where((x >= shift(x)) * (x > shift(x, -1)))[0])


def argclosest(x, y):
    return np.argmin(mag(x - y))


def vector_angle(v1, v2, t):
    v1 = norm(v1)
    v2 = norm(v2)

    s = np.dot(np.cross(v1, v2), t)
    c = np.dot(v1, v2)

    return np.arctan2(s, c)


def trace_line(path, thickness=0.05, sides=15):
    num_points = len(path)
    T = norm(D(path))

    N = [np.eye(3)[T[0].np.argmin()]]

    for t in T:
        N.append(norm(N[-1] - np.dot(N[-1], t) * t))

    qq = norm(N[-1] - np.dot(N[-1], T[0]) * T[0])  # Wrap around to measure angle

    N = N[1:]

    angle = vector_angle(N[0], qq, T[0])

    B = np.cross(T, N)

    theta = np.arange(num_points) * angle / num_points
    c = np.cos(theta).reshape((num_points, 1))
    s = np.sin(theta).reshape((num_points, 1))

    N, B = c * N - s * B, s * N + c * B

    phi = 2 * np.pi * np.arange(sides) / sides
    phi.shape += (1,)
    x = np.sin(phi)
    y = np.cos(phi)

    # normals = -np.vstack([n * x + b * y for n, b in zip(N, B)])
    points = np.vstack([p + thickness * (n * x + b * y) for p, n, b in zip(path, N, B)])

    return make_tube(points, sides)


if __name__ == '__main__':
    import sys

    m = Mesh(sys.argv[1])
    x, y, z = m.points.T

    print('Points: %d' % len(m.points))
    print('Triangles: %d' % len(m.triangles))

    # m.triangles[0] = m.triangles[0, ::-1] #Flip a triangle to test the closed shape detection
    print('Volume: %f (%s)' % (m.volume(), 'mesh appears closed and properly oriented' if m.is_closed(
        tol=1E-12) else 'MESH DOES NOT APPEAR CLOSED AND ORIENTED PROPERLY)\n   (A translated copy had a different'
                        ' calculated volume at 1 part in 10^12;\n    this could be a rounding error.'))
    print('Print price: $%.2f (assuming units=mm and $0.30/cc)' % (m.volume() / 1000. * 0.30))
    print('X extents: (%f, %f)' % (min(x), max(x)))
    print('Y extents: (%f, %f)' % (min(y), max(y)))
    print('Z extents: (%f, %f)' % (min(z), max(z)))

    from enthought.mayavi import mlab

    mlab.triangular_mesh(x, y, z, m.triangles)
    mlab.show()
