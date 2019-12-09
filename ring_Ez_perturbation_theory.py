from __future__ import division

import meep as mp
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


def main():
    n = 3.4                 # index of waveguide
    w = 1                   # width of waveguide
    r = 1
    a = r                   # inner radius of ring
    b = a + w               # outer radius of ring
    pad = 4                 # padding between waveguide and edge of PML

    dpml = 2                # thickness of PML
    pml_layers = [mp.PML(dpml)]

    resolution = 10

    sxy = 2 * (b + pad + dpml)            # radial size (cell is from 0 to sr)
    cell = mp.Vector3(sxy, sxy, 0)

    c1 = mp.Cylinder(radius=b, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=a)
    geometry = [c1, c2]

    # Finding a resonance mode with a high Q-value (calculated with Harminv)

    fcen = 0.15         # pulse center frequency
    df = 0.1            # pulse width (in frequency)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        symmetries=[mp.Mirror(mp.Y)],
                        boundary_layers=pml_layers)

    h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=200)

    Harminv_Q_values = [mode.freq for mode in h.modes]
    max_Q_index = np.argmax(Harminv_Q_values)
    Harminv_freq_at_R = h.modes[max_Q_index].freq
    print(f'The selected resonance frequency to investigate is {Harminv_freq_at_R}')
    sim.reset_meep()

    resolution = 50

    fcen = Harminv_freq_at_R
    df = 0.01

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        symmetries=[mp.Mirror(mp.Y)],
                        boundary_layers=pml_layers)

    sim.run(until_after_sources=200)

    npts_inner = 10
    angles_inner = 2 * np.pi / npts_inner * np.arange(npts_inner)
    deps_inner = 1 - n ** 2
    parallel_fields_inner = []
    for angle in angles_inner:
        point = mp.Vector3(a * np.cos(angle), a * np.sin(angle))
        e_z_field = abs(sim.get_field_point(mp.Ez, point))
        e_total_field = deps_inner * e_z_field
        parallel_fields_inner.append(e_total_field)

    npts_outer = npts_inner
    angles_outer = 2 * np.pi / npts_outer * np.arange(npts_outer)
    deps_outer = n**2 - 1
    parallel_fields_outer = []
    for angle in angles_outer:
        point = mp.Vector3(b * np.cos(angle), b * np.sin(angle))
        e_z_field = abs(sim.get_field_point(mp.Ez, point))
        e_total_field = deps_outer * e_z_field
        parallel_fields_outer.append(e_total_field)

    numerator_surface_integral = 2 * np.pi * b * mean(mean(parallel_fields_inner), mean(parallel_fields_outer))
    denominator_surface_integral = sim.electric_energy_in_box(center=mp.Vector3(), size=mp.Vector3(sxy-dpml, sxy-dpml))
    perturb_theory_dw_dR = -Harminv_freq_at_R * numerator_surface_integral / (4 * denominator_surface_integral)

    center_diff_dw_dR = []
    Harminv_freqs_at_R_plus_dR = []

    drs = np.logspace(start=-3, stop=-1, num=10)

    for dr in drs:
        sim.reset_meep()
        w = 1 + dr  # width of waveguide
        b = a + w
        print(f'The current dr is dr={dr}')
        if len(Harminv_freqs_at_R_plus_dR) == 0:
            fcen = Harminv_freq_at_R
        else:
            fcen = Harminv_freqs_at_R_plus_dR[-1]
        df = 0.01

        sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))]

        c1 = mp.Cylinder(radius=b, material=mp.Medium(index=n))
        c2 = mp.Cylinder(radius=a)
        geometry = [c1, c2]

        sim = mp.Simulation(cell_size=cell,
                            geometry=geometry,
                            sources=sources,
                            resolution=resolution,
                            symmetries=[mp.Mirror(mp.Y)],
                            boundary_layers=pml_layers)

        h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
        sim.run(mp.after_sources(h), until_after_sources=200)

        Harminv_freq_at_R_plus_dR = h.modes[0].freq
        Harminv_freqs_at_R_plus_dR.append(Harminv_freq_at_R_plus_dR)

        dw_dR = (Harminv_freq_at_R_plus_dR - Harminv_freq_at_R) / dr
        center_diff_dw_dR.append(dw_dR)

    relative_errors_dw_dR = [abs((dw_dR - perturb_theory_dw_dR) / dw_dR) for dw_dR in center_diff_dw_dR]

    perturb_predicted_freqs_at_R_plus_dR = [dr * perturb_theory_dw_dR + Harminv_freq_at_R for dr in drs]
    relative_errors_freqs_at_R_plus_dR = [abs((perturb_predicted_freqs_at_R_plus_dR[i] - Harminv_freqs_at_R_plus_dR[i]) / Harminv_freqs_at_R_plus_dR[i]) for i in range(len(Harminv_freqs_at_R_plus_dR))]

    if mp.am_master():
        plt.figure(dpi=150)
        plt.loglog(drs, relative_errors_dw_dR, 'bo-', label='relative error')
        plt.grid(True, which='both', ls='-')
        plt.xlabel('perturbation amount $dr$')
        plt.ylabel('relative error between $dω/dR$')
        plt.legend(loc='upper right')
        plt.title('Comparison of Perturbation Theory and \nCenter-Difference Calculations in Finding $dω/dR$')
        plt.tight_layout()
        # plt.show()
        plt.savefig('ring_Ez_perturbation_theory.dw_dR_error.png')
        plt.clf()

        plt.figure(dpi=150)
        plt.loglog(drs, relative_errors_freqs_at_R_plus_dR, 'bo-', label='relative error')
        plt.grid(True, which='both', ls='-')
        plt.xlabel('perturbation amount $dr$')
        plt.ylabel('relative error between $ω(R+dR)$')
        plt.legend(loc='upper left')
        plt.title('Comparison of resonance frequencies at $R+dR$ predicted by\nperturbation theory and found with Harminv')
        plt.tight_layout()
        # plt.show()
        plt.savefig('ring_Ez_perturbation_theory.freqs_error.png')
        plt.clf()


if __name__ == '__main__':
    main()