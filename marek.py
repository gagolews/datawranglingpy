"""
Basic, vectorised tools missing in numpy/scipy/pandas but available in R.

Copyleft (C) 2015-2022, Marek Gagolewski <https://www.gagolewski.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License
Version 3, 19 November 2007, published by the Free Software Foundation.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License Version 3 for more details.
You should have received a copy of the License along with this program.
If this is not the case, refer to <https://www.gnu.org/licenses/>.
"""

import itertools
import numpy as np


def print_labelled_array(l, v):
    """
    Pretty-prints a numpy array v with labelled axes.

    Parameters
    ----------

        l : list of list of str
            Labels for all the axes' levels;
            a list of length v.ndim with l[i] being a
            list of v.shape[i] strings.
            Alternatively, if v.ndim == 1,
            it can be a list of v.shape[0] strings.

        v : ndarray
            The array to be pretty-printed.

    Examples
    --------

    print_labelled_array(["a", "b", "c"], [3.14, -24, 29])
    ##      a     b     c
    ##   3.14 -24.0  29.0

    print_labelled_array(
        (["XX", "YY"], ["a", "b", "c"]),
        np.arange(6).reshape(2, 3)
    )
    ##      a  b  c
    ##  XX  0  1  2
    ##  YY  3  4  5
    print_labelled_array(
        (["XX", "YY"], ["f"], ["a", "b", "c"],
         ["spam", "bacon", "eggs", "rye"]),
        np.arange(24).reshape(2, 1, 3, 4)
    )
    ##     XX, f, :, :
    ##             spam bacon  eggs   rye
    ##         a     0     1     2     3
    ##         b     4     5     6     7
    ##         c     8     9    10    11
    ##     YY, f, :, :
    ##             spam bacon  eggs   rye
    ##         a    12    13    14    15
    ##         b    16    17    18    19
    ##      c    20    21    22    23
    """
    v = np.array(v)
    if v.ndim == 1 and type(l[0]) in [str, np.str_]:
        l = [l]  # for convenience

    if len(l) != v.ndim:
        raise Exception("len(l) != v.ndim")
    for i in range(len(l)):
        if len(l[i]) != v.shape[i]:
            raise Exception("len(l[i]) != v.shape[i]")

    vs = np.atleast_2d(v).astype(str)

    vs = np.insert(vs, 0, l[-1], axis=vs.ndim-2)
    if v.ndim > 1:
        vs = np.insert(vs, 0, np.append("", l[-2]), axis=vs.ndim-1)
    maxlen = np.max(np.char.str_len(vs)+1)
    vs = np.char.rjust(vs, maxlen)

    def __print_mat(vs_cur):
        for i in range(vs_cur.shape[0]):
            for j in range(vs_cur.shape[1]):
                print(vs_cur[i, j], end="")
            print("")

    if vs.ndim == 2:
        __print_mat(vs)
    else:
        for i in itertools.product(*[range(i) for i in vs.shape[:-2]]):
            lev = ""
            for j in range(vs.ndim-2):
                lev += l[j][i[j]] + ", "
                #lev += str(i[j]) + "=" + l[j][i[j]] + ", "
            print(f"{lev}:, :")
            __print_mat(vs[i])
