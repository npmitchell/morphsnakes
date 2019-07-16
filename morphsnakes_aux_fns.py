import os
import logging
import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
import morphsnakes as ms
import basics.h5py_wrapper as dh5
import copy
from polyhedron.polyhedron import Polyhedron
import data_handling as dh
import basics.dataio as dio
import basics.plotting.plotting as bplt
import mcubes
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
import basics.h5py_wrapper as h5

"""
"""


def plot_levelset_result(levelset, img, name='', imdir='./', fig=None, fig2=None, ax=None,
                         title=None, comparison_mesh=None, save=True, show=False, alpha=0.5, thres=1, sz=3):
    """
    
    Parameters
    ----------
    levelset
    name
    ax
    title : str or None
    comparison_mesh
    save
    imdir

    Returns
    -------

    """
    if fig is None:
        fig = plt.figure(1)
    else:
        fig.clf()

    if fig2 is None:
        fig2 = plt.figure(2)
    else:
        fig2.clf()

    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    if img is not None:
        # Also show cross sections of the volume
        ax2 = fig2.add_subplot(111)
        nx = int(np.shape(img)[0] * 0.5)
        ny = int(np.shape(img)[1] * 0.5)
        nz = int(np.shape(img)[2] * 0.5)
        xslice = img[nx, :, :]
        yslice = img[:, ny, :]
        zslice = img[:, :, nz]

    coords, triangles = mcubes.marching_cubes(levelset, 0.5)

    # Plot the level set
    ax.set_aspect('equal')
    ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],  # change axes!!!!
                    triangles=triangles, alpha=alpha)

    if comparison_mesh is not None:
        mm = comparison_mesh
        ax.plot_trisurf(mm.points[:, 1], mm.points[:, 0], mm.points[:, 2],
                        triangles=mm.triangles, alpha=0.3)
        ax.set_xlabel(r'x [$\mu$m]')
        ax.set_ylabel(r'y [$\mu$m]')
        ax.set_zlabel(r'z [$\mu$m]')

    if title is None:
        title = 'Morphological Chan-Vese level set'
    ax.set_title(title)

    if save:
        imdir = dio.prepdir(imdir)
        dio.ensure_dir(imdir)
        bplt.set_axes_equal(ax)
        # ax.view_init(0, 30)
        # imfn = imdir + '{0:06d}'.format(counter[0]) + '.png'
        # print 'saving ', imfn
        # fig.savefig(imfn)
        ax.view_init(0, 0)
        imfn = imdir + name + '_viewx' + '.png'
        print('saving ', imfn)
        fig.savefig(imfn)
        ax.view_init(0, 90)
        imfn = imdir + name + '_viewy' + '.png'
        print('saving ', imfn)
        fig.savefig(imfn)
        ax.view_init(-90, 90)
        imfn = imdir + name + '_viewz' + '.png'
        print('saving ', imfn)
        fig.savefig(imfn)

        # plt.show()
        # plt.close('all')
        ax.cla()
        # raise RuntimeError
    elif show:
        plt.show()

    # Also show 2d slices of the 3D volume
    if img is not None:
        slices = [xslice.T, yslice.T, zslice.T]
        lsxpts = np.where(np.abs(coords[:, 0] - nx) < thres)[0]
        lsypts = np.where(np.abs(coords[:, 1] - ny) < thres)[0]
        lszpts = np.where(np.abs(coords[:, 2] - nz) < thres)[0]
        for (slice, ii) in zip(slices, range(len(slices))):
            ax2.imshow(slice)

            # Show the comparison mesh boundary
            ax2.set_aspect('equal')

            # Show the level set
            if ii == 0:
                ax2.scatter(coords[lsxpts, 1], coords[lsxpts, 2], s=sz, alpha=alpha)
                ax2.set_xlabel('y')
                ax2.set_ylabel('z')
            elif ii == 1:
                ax2.scatter(coords[lsypts, 0], coords[lsypts, 2], s=sz, alpha=alpha)
                ax2.set_xlabel('x')
                ax2.set_ylabel('z')
            elif ii == 2:
                ax2.scatter(coords[lszpts, 0], coords[lszpts, 1], s=sz, alpha=alpha)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')

            ax2.xaxis.set_ticks([])
            ax2.yaxis.set_ticks([])
            plt.text(0.5, 0.9, title, va='center', ha='center', transform=fig2.transFigure)

            if save:
                dio.ensure_dir(imdir)
                ax2.set_aspect('equal')
                imfn = imdir + name + '_slice{0:01d}.png'.format(ii)
                print('saving ', imfn)
                fig2.savefig(imfn)
                # plt.show()
                # plt.close('all')
                ax2.cla()
                # raise RuntimeError
            elif show:
                plt.show()


def dilate(u):
    """Dilate the implicit surface (ie a segmentation)"""
    u = ndi.binary_dilation(u)
    return u


def save_ls_as_h5(filename, ls):
    """

    Parameters
    ----------
    filename : str
        the pathname of the hdf5 file to save
    ls : ndarray int or bool
        the implicit level set: an ND array with ones inside and zeros outside

    Returns
    -------
    f : pointer
        a pointer to the file on disk
    """
    # Create the hdf5 file and/or add data to the file
    return h5.write(ls, overwrite=True, filename=filename, key='implicit_levelset')

