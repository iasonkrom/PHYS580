import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class Orbit:
    """
    Simulates the motion of a charged particle in magnetic and optional electric fields.

    Parameters
    ----------
    m : float
        Mass of the particle.
    R0 : float
        Reference radius for the magnetic field.
    B0 : float
        Magnetic field strength at the reference radius.
    Bp0 : float
        Poloidal magnetic field strength.
    q : float
        Charge of the particle.
    tf : float
        Final time for the simulation.
    v0 : float
        Initial velocity of the particle.
    R : float
        Initial radial position of the particle.
    phi : float
        Initial azimuthal angle of the particle.
    Z : float
        Initial vertical position of the particle.
    v_prev : ndarray
        Initial velocity vector of the particle.
    E_field : float, optional
        Strength of the electric field (default is None, meaning no electric field).
    dt_factor : float, optional
        Factor to scale the time step by (default is 0.05).
        The time step is calculated as dt_factor / omega, where omega is the cyclotron frequency of the particle.

    Attributes
    ----------
    omega : float
        Cyclotron frequency of the particle.
    dt : float
        Time step for the simulation.
    v_pprev : float
        Velocity component in the azimuthal direction in the previous step.
    N : int
        Number of steps in the simulation.

    Methods
    -------
    simulate_motion()
        Runs the particle motion simulation and returns the trajectory and other data.
    """

    def __init__(
        self,
        m,
        R0,
        B0,
        Bp0,
        q,
        tf,
        v0,
        R,
        phi,
        Z,
        v_prev,
        E_field=None,
        dt_factor=0.05,
    ):
        self.m = m
        self.R0 = R0
        self.B0 = B0
        self.Bp0 = Bp0
        self.q = q
        self.tf = tf
        self.v0 = v0
        self.R = R
        self.phi = phi
        self.Z = Z
        self.v_prev = v_prev

        if E_field is not None:
            self.E_field = E_field
        else:
            self.E_field = 0

        self.omega = (self.q / self.m) * self.B0
        self.dt = dt_factor / self.omega
        self.v_pprev = self.v_prev[2]
        self.N = int(self.tf / self.dt)

    def _perpproj(self, A, B):
        """
        Calculates the perpendicular projection of vector A onto plane perpendicular to vector B.

        Parameters
        ----------
        A : ndarray
            The vector to be projected.
        B : ndarray
            The vector defining the plane of projection.

        Returns
        -------
        ndarray
            The projection of A onto the plane perpendicular to B.
        """
        return A - np.dot(A, B) / np.linalg.norm(B) ** 2 * B

    def _calculate_E(self, R, Z, R0, E_field):
        """
        Calculates the electric field vector at a given position.

        Parameters
        ----------
        R : float
            Radial position where the electric field is calculated.
        Z : float
            Vertical position where the electric field is calculated.
        R0 : float
            Reference radius for the magnetic field.
        E_field : float
            Strength of the electric field.

        Returns
        -------
        ndarray
            The electric field vector at the given position.
        """
        r = np.sqrt((R - R0) ** 2 + Z**2)
        Er = E_field * (R - R0) / r
        Ephi = 0
        Ez = E_field * Z / r
        return np.array([Er, Ephi, Ez])

    def _calculate_B(self, R, Z, R0, B0, Bp0):
        """
        Calculates the magnetic field vector at a given position.

        Parameters
        ----------
        R : float
            Radial position where the magnetic field is calculated.
        Z : float
            Vertical position where the magnetic field is calculated.
        R0 : float
            Reference radius for the magnetic field.
        B0 : float
            Magnetic field strength at the reference radius.
        Bp0 : float
            Poloidal magnetic field strength.

        Returns
        -------
        tuple
            A tuple containing the magnetic field vector and the radial distance from the reference radius.
        """
        r = np.sqrt((R - R0) ** 2 + Z**2)
        Br = (Bp0 * R0 * Z) / (r * R)
        Bphi = (B0 * R0) / R
        Bz = (Bp0 * (R0 - R) * R0) / (r * R)
        return np.array([Br, Bphi, Bz]), r

    def _update_velocity(self, v_prev, E, B, Bstar, tau):
        """
        Updates the velocity of the particle based on the Boris algorithm.

        Parameters
        ----------
        v_prev : ndarray
            Previous velocity vector of the particle.
        E : ndarray
            Electric field vector.
        B : ndarray
            Magnetic field vector.
        Bstar : ndarray
            Modified magnetic field vector for the Boris algorithm.
        tau : float
            Time step scaled by particle's charge-to-mass ratio.

        Returns
        -------
        ndarray
            Updated velocity vector of the particle.
        """
        v_ = v_prev + tau * E
        c1 = 4 / (4 + np.linalg.norm(Bstar) ** 2)
        c2 = 2 * c1 - 1
        c3 = (c1 / 2) * np.dot(v_, Bstar)
        v_plus = c1 * np.cross(v_, Bstar) + c2 * v_ + c3 * Bstar
        return v_plus + tau * E

    def simulate_motion(self):
        """
        Runs the particle motion simulation over the specified time frame.

        Returns
        -------
        tuple
            A tuple containing the particle's trajectory, magnetic moment, magnetic field, and electric field (if applicable) over time.
        """
        x = np.empty((3, self.N))
        moment = np.empty(self.N)
        field = np.empty(self.N)
        efield = np.empty(self.N)

        for i in trange(self.N):
            B, r = self._calculate_B(self.R, self.Z, self.R0, self.B0, self.Bp0)
            E = self._calculate_E(self.R, self.Z, self.R0, self.E_field)
            field[i] = np.linalg.norm(B)
            efield[i] = np.linalg.norm(E)
            Bstar = (self.q * self.dt / self.m) * (
                B
                + np.array(
                    [
                        0,
                        0,
                        (self.m / (self.q * self.R))
                        * (1.5 * self.v_prev[1] - 0.5 * self.v_pprev),
                    ]
                )
            )
            v = self._update_velocity(
                self.v_prev, E, B, Bstar, self.q / (2 * self.m) * self.dt
            )
            x[:, i] = (
                np.array([self.R, self.phi, self.Z])
                + (v / np.array([1, self.R, 1])) * self.dt
            )
            self.R, self.phi, self.Z = x[:, i]
            v_perp = self._perpproj(self.v_prev, B)
            moment[i] = self.m * np.linalg.norm(v_perp) ** 2 / (2 * np.linalg.norm(B))
            self.v_pprev = self.v_prev[1] if i > 0 else self.v_pprev
            self.v_prev = v

        if self.E_field != 0:
            return x, moment, field, efield
        return x, moment, field


def plot_criterion(t, moment, field, label=None):
    """
    Plots the trapped/passing criterion for a particle in a magnetic field.

    Parameters
    ----------
    t : ndarray
        Array of time points.
    moment : ndarray
        Array of magnetic moments of the particle at each time point.
    field : ndarray
        Array of magnetic field magnitudes at each time point.
    label : str, optional
        Label for the plot.
    """
    Bmax = np.max(field)
    plt.plot(t, moment * Bmax / 1.602e-13, label=label, lw=1)
    plt.plot(
        t,
        np.mean(moment) * np.ones_like(moment) * Bmax / 1.602e-13,
        linestyle="--",
        lw=1,
    )
    plt.plot(t, 3.5 * np.ones_like(moment), linestyle=":", lw=1)


def plot_top_view(x, label=None):
    """
    Plots the top view (R, φ plane) of the particle's orbit in a polar coordinate system.

    Parameters
    ----------
    x : ndarray
        Array containing the particle's trajectory (R, φ, Z) over time.
    label : str, optional
        Label for the plot.
    """
    R = x[0]
    phi = x[1]
    plt.polar(phi, R, label=label, lw=0.5)


def plot_poloidal_view(x, label=None):
    """
    Plots the poloidal view (R, Z plane) of the particle's orbit.

    Parameters
    ----------
    x : ndarray
        Array containing the particle's trajectory (R, φ, Z) over time.
    label : str, optional
        Label for the plot.
    """
    plt.plot(x[0], x[2], label=label, lw=0.5)
