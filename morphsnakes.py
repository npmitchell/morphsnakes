# -*- coding: utf-8 -*-
from __future__ import print_function
import copy

"""
====================
Morphological Snakes
====================

*Morphological Snakes* [1]_ are a family of methods for image segmentation.
Their behavior is similar to that of active contours (for example, *Geodesic
Active Contours* [2]_ or *Active Contours without Edges* [3]_). However,
*Morphological Snakes* use morphological operators (such as dilation or
erosion) over a binary array instead of solving PDEs over a floating point
array, which is the standard approach for active contours. This makes
*Morphological Snakes* faster and numerically more stable than their
traditional counterpart.

There are two *Morphological Snakes* methods available in this implementation:
*Morphological Geodesic Active Contours* (**MorphGAC**, implemented in the
function ``morphological_geodesic_active_contour``) and *Morphological Active
Contours without Edges* (**MorphACWE**, implemented in the function
``morphological_chan_vese``).

**MorphGAC** is suitable for images with visible contours, even when these
contours might be noisy, cluttered, or partially unclear. It requires, however,
that the image is preprocessed to highlight the contours. This can be done
using the function ``inverse_gaussian_gradient``, although the user might want
to define their own version. The quality of the **MorphGAC** segmentation
depends greatly on this preprocessing step.

On the contrary, **MorphACWE** works well when the pixel values of the inside
and the outside regions of the object to segment have different averages.
Unlike **MorphGAC**, **MorphACWE** does not require that the contours of the
object are well defined, and it works over the original image without any
preceding processing. This makes **MorphACWE** easier to use and tune than
**MorphGAC**.

References
----------

.. [1] A Morphological Approach to Curvature-based Evolution of Curves and
       Surfaces, Pablo Márquez-Neila, Luis Baumela and Luis Álvarez. In IEEE
       Transactions on Pattern Analysis and Machine Intelligence (PAMI),
       2014, DOI 10.1109/TPAMI.2013.106
.. [2] Geodesic Active Contours, Vicent Caselles, Ron Kimmel and Guillermo
       Sapiro. In International Journal of Computer Vision (IJCV), 1997,
       DOI:10.1023/A:1007979827043
.. [3] Active Contours without Edges, Tony Chan and Luminita Vese. In IEEE
       Transactions on Image Processing, 2001, DOI:10.1109/83.902291

"""
__author__ = "P. Márquez Neila <p.mneila@upm.es>"

from itertools import cycle

import numpy as np
from scipy import ndimage as ndi

__all__ = ['morphological_chan_vese',
           'morphological_geodesic_active_contour',
           'inverse_gaussian_gradient',
           'circle_level_set',
           'checkerboard_level_set'
           ]

__version__ = (2, 0, 1)
__version_str__ = ".".join(map(str, __version__))


class _fcycle(object):

    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1


def sup_inf(u):
    """SI operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i))

    return np.array(erosions, dtype=np.int8).max(0)


def inf_sup(u):
    """IS operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i))

    return np.array(dilations, dtype=np.int8).min(0)


_curvop = _fcycle([lambda u: sup_inf(inf_sup(u)),  # SIoIS
                   lambda u: inf_sup(sup_inf(u))])  # ISoSI


def curvop():
    return _fcycle([lambda u: sup_inf(inf_sup(u)),  # SIoIS
                   lambda u: inf_sup(sup_inf(u))])  # ISoSI


def _check_input(image, init_level_set):
    """Check that shapes of `image` and `init_level_set` match."""
    if not image.ndim in [2, 3]:
        raise ValueError("`image` must be a 2 or 3-dimensional array.")

    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("The dimensions of the initial level set do not "
                         "match the dimensions of the image.")


def _init_level_set(init_level_set, image_shape):
    """Auxiliary function for initializing level sets with a string.

    If `init_level_set` is not a string, it is returned as is.
    """
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = checkerboard_level_set(image_shape)
        elif init_level_set == 'circle':
            res = circle_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in "
                             "['checkerboard', 'circle']")
    else:
        res = init_level_set
    return res


def circle_level_set(image_shape, center=None, radius=None):
    """Create a circle or (hyper)sphere level set with binary values.

    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image
    center : tuple of positive integers, optional
        Coordinates of the center of the circle given in (row, column). If not
        given, it defaults to the center of the image.
    radius : float, optional
        Radius of the circle. If not given, it is set to the 75% of the
        smallest image dimension.

    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the circle with the given `radius` and `center`.

    See also
    --------
    checkerboard_level_set
    """

    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid) ** 2, 0))
    res = np.int8(phi > 0)
    if np.sum(res.ravel()) == 0:
        raise RuntimeError('The initial level set has no intersection with the data volume')
    
    return res


def checkerboard_level_set(image_shape, square_size=5):
    """Create a checkerboard level set with binary values.

    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image.
    square_size : int, optional
        Size of the squares of the checkerboard. It defaults to 5.

    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the checkerboard.

    See also
    --------
    circle_level_set
    """

    grid = np.ogrid[[slice(i) for i in image_shape]]
    grid = [(grid_i // square_size) & 1 for grid_i in grid]

    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    """Inverse of gradient magnitude.

    Compute the magnitude of the gradients in the image and then inverts the
    result in the range [0, 1]. Flat areas are assigned values close to 1,
    while areas close to borders are assigned values close to 0.

    This function or a similar one defined by the user should be applied over
    the image as a preprocessing step before calling
    `morphological_geodesic_active_contour`.

    Parameters
    ----------
    image : (M, N) or (L, M, N) array
        Grayscale image or volume.
    alpha : float, optional
        Controls the steepness of the inversion. A larger value will make the
        transition between the flat areas and border areas steeper in the
        resulting array.
    sigma : float, optional
        Standard deviation of the Gaussian filter applied over the image.

    Returns
    -------
    gimage : (M, N) or (L, M, N) array
        Preprocessed image (or volume) suitable for
        `morphological_geodesic_active_contour`.
    """
    gradnorm = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def morphological_chan_vese(image, iterations, init_level_set='checkerboard',
                            smoothing=1, lambda1=1, lambda2=1, nu=None, post_smoothing=0,
                            post_nu=None, iter_callback=lambda x: None, exit_thres=None):
    """Morphological Active Contours without Edges (MorphACWE)

    Active contours without edges implemented with morphological operators. It
    can be used to segment objects in images and volumes without well defined
    borders. It is required that the inside of the object looks different on
    average than the outside (i.e., the inner area of the object should be
    darker or lighter than the outer area on average).

    Parameters
    ----------
    image : (M, N) or (L, M, N) array
        Grayscale image or volume to be segmented.
    iterations : uint
        Number of iterations to run
    init_level_set : str, (M, N) array, or (L, M, N) array
        Initial level set. If an array is given, it will be binarized and used
        as the initial level set. If a string is given, it defines the method
        to generate a reasonable initial level set with the shape of the
        `image`. Accepted values are 'checkerboard' and 'circle'. See the
        documentation of `checkerboard_level_set` and `circle_level_set`
        respectively for details about how these level sets are created.
    nu : float, optional
        If not None and nonzero, applies pressure to the surface. If negative,
        applies negative pressure at each iteration or every 1/nu iterations
        if |nu|<1. In other words, nu is the number of times to apply a
        dilation or erosion at each timestep.
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    lambda1 : float, optional
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
    lambda2 : float, optional
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    post_smoothing : int, optional
        Number of iterations for smoothing after exiting iterative procedure
    post_nu : int, optional
        If not None and nonzero, applies a dilation or erosion post_nu times to
        the resulting level set.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.
    exit_thres : float or None
        If given, we truncate the algorithm before #iterations = iterations if
        the segmentation u differs by less than the given threshold, expressed
        as a fraction of the total volume.

    Returns
    -------
    out : (M, N) or (L, M, N) array
        Final segmentation (i.e., the final level set)

    See also
    --------
    circle_level_set, checkerboard_level_set

    Notes
    -----

    This is a version of the Chan-Vese algorithm that uses morphological
    operators instead of solving a partial differential equation (PDE) for the
    evolution of the contour. The set of morphological operators used in this
    algorithm are proved to be infinitesimally equivalent to the Chan-Vese PDE
    (see [1]_). However, morphological operators are do not suffer from the
    numerical stability issues typically found in PDEs (it is not necessary to
    find the right time step for the evolution), and are computationally
    faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """

    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    u = np.int8(init_level_set > 0)

    iter_callback(u)
    kk = 0
    done = False
    u_prev = copy.deepcopy(u)
    while not done:
        # First apply pressure
        if nu is not None:
            if nu > 0:
                # If nu is an integer 1 or higher, apply it each time, otherwise if less than 1 apply it every so often
                if nu % 1 == 0 or nu > 1:
                    for _ in range(int(nu)):
                        u = ndi.binary_dilation(u)

                    u = u.astype(int)
                else:
                    if nu * kk % 1 < nu:
                        u = ndi.binary_dilation(u)
                        u = u.astype(int)
            elif nu < 0:
                # If nu is an integer -1 or lower, apply it each time
                if nu % 1 == 0 or nu < -1:
                    for _ in range(int(-nu)):
                        u = ndi.binary_erosion(u)

                    u = u.astype(int)
                else:
                    if -nu * kk % 1 < -nu:
                        u = ndi.binary_erosion(u)
                        u = u.astype(int)

        # inside = u > 0
        # outside = u <= 0
        # Weight the image with the total amount of intensity outside u
        # numerator: Integral[ image * (1 - u) ] dV = 0 if image is BW and u perfectly segments, but > 0 if image has
        # nonzero values outside object or if u segments less than the object.
        # denominator: total volume outside u
        c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-8)
        # Weight the image with the total amount of intensity inside u
        # numerator: Integral[ image * u ] dV = Volume of segmented u if u segments BW image.
        c1 = (image * u).sum() / float(u.sum() + 1e-8)

        # Image attachment
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1 * (image - c1) ** 2 - lambda2 * (image - c0) ** 2)

        u[aux < 0] = 1
        u[aux > 0] = 0

        # Smoothing
        # If smoothing is an integer 1 or higher, apply it each time, otherwise if less than 1 apply it every so often
        if smoothing % 1 == 0 or smoothing > 1:
            for _ in range(int(smoothing)):
                u = _curvop(u)
        else:
            if smoothing * kk % 1 < smoothing:
                u = _curvop(u)

        # Run the callback
        iter_callback(u, u_prev)

        # Check if we have converged, if a convergence threshold is given
        if exit_thres is not None:
            if np.sum(u) > 0:
                frac_change = float(np.sum(np.abs(u - u_prev).ravel())) / float(np.sum(u))
                if frac_change < exit_thres:
                    done = True
            else:
                done = True
                print('WARNING: level set has shrunk to zero size!')

            u_prev2 = copy.deepcopy(u_prev)

            # Also compare against the time point before the last in case of flip flopping
            if kk > 0:
                frac_change2 = float(np.sum(np.abs(u - u_prev2).ravel())) / float(np.sum(u))
                if frac_change2 < exit_thres:
                    done = True
                msg = '{0:3d}'.format(kk) + ': fractional change = {0:0.9f}'.format(min(frac_change, frac_change2))
                print(msg, end='\r')

            u_prev = copy.deepcopy(u)

        # Update iteration number and possibly exit if done with #iterations
        kk += 1
        if kk == iterations:
            done = True

    # Output semifinal file/image of the evolution
    iter_callback(u, u_prev, force=True)

    if post_nu is not None:
        if post_nu > 0:
            for _ in range(int(post_nu)):
                u = ndi.binary_dilation(u)
        elif post_nu < 0:
            for _ in range(int(-post_nu)):
                u = ndi.binary_erosion(u)

    for _ in range(post_smoothing):
        u = _curvop(u)

    # Output final file/image of the evolution
    iter_callback(u, u_prev, force=True)

    return u


def morphological_geodesic_active_contour(gimage, iterations,
                                          init_level_set='circle', smoothing=1,
                                          threshold='auto', balloon=0,
                                          iter_callback=lambda x: None):
    """Morphological Geodesic Active Contours (MorphGAC).

    Geodesic active contours implemented with morphological operators. It can
    be used to segment objects with visible but noisy, cluttered, broken
    borders.

    Parameters
    ----------
    gimage : (M, N) or (L, M, N) array
        Preprocessed image or volume to be segmented. This is very rarely the
        original image. Instead, this is usually a preprocessed version of the
        original image that enhances and highlights the borders (or other
        structures) of the object to segment.
        `morphological_geodesic_active_contour` will try to stop the contour
        evolution in areas where `gimage` is small. See
        `morphsnakes.inverse_gaussian_gradient` as an example function to
        perform this preprocessing. Note that the quality of
        `morphological_geodesic_active_contour` might greatly depend on this
        preprocessing.
    iterations : uint
        Number of iterations to run.
    init_level_set : str, (M, N) array, or (L, M, N) array
        Initial level set. If an array is given, it will be binarized and used
        as the initial level set. If a string is given, it defines the method
        to generate a reasonable initial level set with the shape of the
        `image`. Accepted values are 'checkerboard' and 'circle'. See the
        documentation of `checkerboard_level_set` and `circle_level_set`
        respectively for details about how these level sets are created.
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.

    Returns
    -------
    out : (M, N) or (L, M, N) array
        Final segmentation (i.e., the final level set)

    See also
    --------
    inverse_gaussian_gradient, circle_level_set, checkerboard_level_set

    Notes
    -----

    This is a version of the Geodesic Active Contours (GAC) algorithm that uses
    morphological operators instead of solving partial differential equations
    (PDEs) for the evolution of the contour. The set of morphological operators
    used in this algorithm are proved to be infinitesimally equivalent to the
    GAC PDEs (see [1]_). However, morphological operators are do not suffer
    from the numerical stability issues typically found in PDEs (e.g., it is
    not necessary to find the right time step for the evolution), and are
    computationally faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """

    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    if threshold == 'auto':
        threshold = np.percentile(image, 40)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    for _ in range(iterations):

        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

    return u


def dilate_n_smooth_m(u, n=1, m=1):
    """Dilate the implicit surface n times, then smooth m times.

    Parameters
    ----------
    u :
    n : int
        number of times to dilate
    m : int
        number of times to smooth

    Returns
    -------
    u
    """
    for _ in range(int(n)):
        u = ndi.binary_dilation(u)

    for _ in range(int(m)):
        u = _curvop(u)

    return u
