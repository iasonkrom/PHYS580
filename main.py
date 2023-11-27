import matplotlib.pyplot as plt
import numpy as np

from orbit import Orbit, plot_criterion, plot_poloidal_view, plot_top_view

if __name__ == "__main__":
    # Question 3

    m = 6.644e-27
    R0 = 1
    B0 = 5
    Bp0 = 1
    q = 2 * 1.602e-19
    omega = (q / m) * B0
    dt_factor = 0.05
    dt = dt_factor / omega
    v0 = np.sqrt(2 * (3.5 * 1.602e-13) / m)
    tf = 100 * (2 * np.pi / omega)
    N = int(tf / dt)
    R, phi, Z = 1.85 * R0, 0, 0
    vr, vz, vphi = 0, -0.6 * v0, 0.8 * v0
    v_prev = np.array([vr, vphi, vz])

    orbit1 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=R,
        phi=phi,
        Z=Z,
        v_prev=v_prev,
        dt_factor=dt_factor,
    )

    t = np.linspace(0, tf, N)
    x, moment, field = orbit1.simulate_motion()

    # Trapped/Passing Criterion Visualization
    plt.figure()
    plot_criterion(t, moment, field, "Particle 1")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [MeV]")
    plt.title("Trapped/Passing Criterion Visualization - Particle 1")
    plt.legend([r"$\mu B_{max}$", "$\\langle\\mu\\rangle B_{max}$", "Total Energy"])
    plt.savefig("particle1_trapped_passing_criterion.pdf", bbox_inches="tight")
    plt.show()

    # Poloidal view of the Particle Orbits ((R,Z) Plane)
    plt.figure()
    plot_poloidal_view(x)
    plt.plot(R0, 0, "o", label="Magnetic axis", markersize=2)
    plt.title("Poloidal view of the Particle Orbit ((R,Z) Plane)")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.savefig("particle1_poloidal_view.pdf", bbox_inches="tight")
    plt.show()

    # Question 4

    m = 6.644e-27
    R0 = 1
    B0 = 5
    Bp0 = 1
    q = 2 * 1.602e-19
    omega = (q / m) * B0
    dt = 0.05 / omega
    v0 = np.sqrt(2 * (3.5 * 1.602e-13) / m)
    tf = 160 * (2 * np.pi / omega)
    N = int(tf / dt)
    R, phi, Z = 1.4 * R0, 0, 0
    vr, vz, vphi = 0, -0.8 * v0, -0.6 * v0
    v_prev = np.array([vr, vphi, vz])

    orbit2 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=R,
        phi=phi,
        Z=Z,
        v_prev=v_prev,
    )

    # Simulation
    t = np.linspace(0, tf, N)
    x, moment, field = orbit2.simulate_motion()

    # Trapped/Passing Criterion Visualization
    plt.figure()
    plot_criterion(t, moment, field, "Particle 1")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [MeV]")
    plt.title("Trapped/Passing Criterion Visualization - Particle 1")
    plt.legend([r"$\mu B_{max}$", "$\\langle\\mu\\rangle B_{max}$", "Total Energy"])
    plt.xlim([0, 4e-6])
    plt.savefig("particle2_trapped_passing_criterion.pdf", bbox_inches="tight")
    plt.show()

    # Poloidal view of the Particle Orbits ((R,Z) Plane)
    plt.figure()
    plot_poloidal_view(x)
    plt.plot(R0, 0, "o", label="Magnetic axis", markersize=2)
    plt.title("Poloidal view of the Particle Orbit ((R,Z) Plane)")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.savefig("particle_poloidal_view.pdf", bbox_inches="tight")
    plt.show()

    # Question 5

    # Parameters common to both particles
    m = 6.644e-27
    R0 = 1
    B0 = 5
    Bp0 = 1
    q = 2 * 1.602e-19
    omega = (q / m) * B0
    dt = 0.05 / omega
    v0 = np.sqrt(2 * (3.5 * 1.602e-13) / m)
    tf = 155 * (2 * np.pi / omega)

    # Create Orbit instances for Particle 1 and Particle 2
    particle1 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.85 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, 0.8 * v0, -0.6 * v0]),
    )
    particle2 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.4 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, -0.6 * v0, -0.8 * v0]),
    )

    # Simulate motion
    x1, moment1, field1 = particle1.simulate_motion()
    x2, moment2, field2 = particle2.simulate_motion()

    t = np.linspace(0, tf, len(moment1))

    # Trapped/Passing Criterion Visualization
    plt.figure()
    plot_criterion(t, moment1, field1, "Particle 1")
    plot_criterion(t, moment2, field2, "Particle 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [MeV]")
    plt.title("Trapped/Passing Criterion Visualization")
    plt.legend()
    plt.xlim([0, 4e-6])
    plt.savefig(
        "both_particles_trapped_passing_criterion_no_efield.pdf", bbox_inches="tight"
    )
    plt.show()

    # Top View of the Particle Orbits ((R,φ) Plane)
    plt.figure()
    plot_top_view(x1, "Particle 1")
    plot_top_view(x2, "Particle 2")
    theta = np.linspace(0, 2 * np.pi, 200)
    R3 = np.full_like(theta, R0)
    plt.polar(theta, R3, "r", label="Magnetic Axis", lw=0.5)
    plt.title("Top View of the Particle Orbits ((R,φ) Plane) - No E-field")
    plt.legend()
    plt.ylim(0, 3)
    plt.savefig("both_particles_top_view_no_efield.pdf", bbox_inches="tight")
    plt.show()

    # Poloidal view of the Particle Orbits ((R,Z) Plane)
    plt.figure()
    plot_poloidal_view(x1, "Particle 1")
    plot_poloidal_view(x2, "Particle 2")
    plt.plot(R0, 0, "o", label="Magnetic Axis", markersize=2)
    plt.title("Poloidal view of the Particle Orbits ((R,Z) Plane) - No E-field")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.savefig("both_particles_poloidal_view_no_efield.pdf", bbox_inches="tight")
    plt.show()

    # Question 5 - E-field

    # Parameters common to both particles
    m = 6.644e-27
    R0 = 1
    B0 = 5
    Bp0 = 1
    q = 2 * 1.602e-19
    omega = (q / m) * B0
    dt = 0.05 / omega
    v0 = np.sqrt(2 * (3.5 * 1.602e-13) / m)
    E_field = 700e3
    tf = 160 * (2 * np.pi / omega)

    # Create Orbit instances for Particle 1 and Particle 2
    particle1 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.85 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, 0.8 * v0, -0.6 * v0]),
        E_field=E_field,
    )
    particle2 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.4 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, -0.6 * v0, -0.8 * v0]),
        E_field=E_field,
    )

    # Simulate motion
    x1, moment1, field1, efield1 = particle1.simulate_motion()
    x2, moment2, field2, efield2 = particle2.simulate_motion()

    t = np.linspace(0, tf, len(moment1))

    # Trapped/Passing Criterion Visualization
    plt.figure()
    plot_criterion(t, moment1, field1, "Particle 1")
    plot_criterion(t, moment2, field2, "Particle 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [MeV]")
    plt.title("Trapped/Passing Criterion Visualization")
    plt.legend()
    plt.xlim([0, 4e-6])
    plt.savefig(
        "both_particles_trapped_passing_criterion_with_efield.pdf", bbox_inches="tight"
    )
    plt.show()

    # Top View of the Particle Orbits ((R,φ) Plane)
    plt.figure()
    plot_top_view(x1, "Particle 1")
    plot_top_view(x2, "Particle 2")
    theta = np.linspace(0, 2 * np.pi, 200)
    R3 = np.full_like(theta, R0)
    plt.polar(theta, R3, "r", label="Magnetic Axis", lw=0.5)
    plt.title(
        "Top View of the Particle Orbits ((R,φ) Plane - $E_0= 700 \\frac{kV}{m}$)"
    )
    plt.legend()
    plt.ylim(0, 3)
    plt.savefig("both_particles_top_view_with_efield.pdf", bbox_inches="tight")
    plt.show()

    # Poloidal view of the Particle Orbits ((R,Z) Plane)
    plt.figure()
    plot_poloidal_view(x1, "Particle 1")
    plot_poloidal_view(x2, "Particle 2")
    plt.plot(R0, 0, "o", label="Magnetic Axis", markersize=2)
    plt.title(
        "Poloidal view of the Particle Orbits ((R,φ) Plane - $E_0= 700 \\frac{kV}{m}$)"
    )
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.savefig("both_particles_poloidal_view_with_efield.pdf", bbox_inches="tight")
    plt.show()

    # Question 6

    # Parameters common to both particles
    m = 6.644e-27
    R0 = 6.2
    B0 = 5
    Bp0 = 1
    q = 2 * 1.602e-19
    omega = (q / m) * B0
    dt = 0.05 / omega
    v0 = np.sqrt(2 * (3.5 * 1.602e-13) / m)
    tf = 1200 * (2 * np.pi / omega)

    # Create Orbit instances for Particle 1 and Particle 2
    particle1 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.85 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, 0.8 * v0, -0.6 * v0]),
    )
    particle2 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.4 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, -0.6 * v0, -0.8 * v0]),
    )

    # Simulate motion
    x1, moment1, field1 = particle1.simulate_motion()
    x2, moment2, field2 = particle2.simulate_motion()

    # Poloidal view of the Particle Orbits ((R,Z) Plane)
    plt.figure()
    plot_poloidal_view(x1, "Particle 1")
    plot_poloidal_view(x2, "Particle 2")
    plt.plot(R0, 0, "o", label="Magnetic Axis", markersize=2)
    plt.title("Poloidal view of the Particle Orbits ((R,Z) Plane) - No E-field")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.savefig("both_particles_ITER_poloidal_view_no_efield.pdf", bbox_inches="tight")
    plt.show()

    # Question 6 - E-field

    # Parameters common to both particles
    m = 6.644e-27
    R0 = 6.2
    B0 = 5
    Bp0 = 1
    q = 2 * 1.602e-19
    omega = (q / m) * B0
    dt = 0.05 / omega
    v0 = np.sqrt(2 * (3.5 * 1.602e-13) / m)
    E_field = 700e3
    tf = 1200 * (2 * np.pi / omega)

    # Create Orbit instances for Particle 1 and Particle 2
    particle1 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.85 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, 0.8 * v0, -0.6 * v0]),
        E_field=E_field,
    )
    particle2 = Orbit(
        m=m,
        R0=R0,
        B0=B0,
        Bp0=Bp0,
        q=q,
        tf=tf,
        v0=v0,
        R=1.4 * R0,
        phi=0,
        Z=0,
        v_prev=np.array([0, -0.6 * v0, -0.8 * v0]),
        E_field=E_field,
    )

    # Simulate motion
    x1, moment1, field1, efield1 = particle1.simulate_motion()
    x2, moment2, field2, efield2 = particle2.simulate_motion()

    # Poloidal view of the Particle Orbits ((R,Z) Plane)
    plt.figure()
    plot_poloidal_view(x1, "Particle 1")
    plot_poloidal_view(x2, "Particle 2")
    plt.plot(R0, 0, "o", label="Magnetic Axis", markersize=2)
    plt.title(
        "Poloidal view of the Particle Orbits ((R,φ) Plane - $E_0= 700 \\frac{kV}{m}$)"
    )
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.savefig(
        "both_particles_ITER_poloidal_view_with_efield.pdf", bbox_inches="tight"
    )
    plt.show()
