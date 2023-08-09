"""
Copyright © 2023 yourname

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np


class Piece:
    """
    Represents a piece in a piecewise-polynomial trajectory
    """

    def __init__(self, degree, duration, coeffs):
        self._degree = degree
        self._duration = duration
        self._coeffs = np.asarray(coeffs, dtype=np.float64)
        if np.shape(self._coeffs) != (3, self._degree + 1):
            raise ValueError("Coefficient matrix must have 3 rows and degree+1 columns")

        # index sequence for raising to power and polynomial derivative coefficients
        self._ind_seq = np.arange(self._degree, -1, -1)

    def __repr__(self):
        return f"""Order: {self._degree}
Duration: {self._duration}
Coefficients: \n{self._coeffs}"""

    @property
    def dim(self):
        return 3

    @property
    def degree(self):
        return self._degree

    @property
    def coeffs(self):
        return self._coeffs

    def get_position(self, time):
        return self._coeffs @ time**self._ind_seq

    def get_velocity(self, time):
        return self._coeffs[:, :-1] @ (time ** self._ind_seq[1:] * self._ind_seq[:-1])

    def get_acceleration(self, time):
        return self._coeffs[:, :-2] @ (
            time ** self._ind_seq[2:] * self._ind_seq[:-2] * self._ind_seq[1:-1]
        )

    @property
    def normalized_pos_coeffs(self):
        return self._coeffs * self._duration**self._ind_seq

    @property
    def normalized_vel_coeffs(self):
        return self._coeffs[:, :-1] * (
            self._duration ** self._ind_seq[:-1] * self._ind_seq[:-1]
        )

    @property
    def normalized_acc_coeffs(self):
        return self._coeffs[:, :-2] * (
            self._duration ** self._ind_seq[:-2]
            * self._ind_seq[:-2]
            * self._ind_seq[1:-1]
        )
