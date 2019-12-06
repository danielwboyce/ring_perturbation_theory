from __future__ import division

import meep as mp
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


def main():
    n = 3.4                 # index of waveguide
    r = 1
    a = r                   # inner radius of ring
    w = 1                   # width of waveguide
    b = a + w               # outer radius of ring
    pad = 4                 # padding between waveguide and edge of PML

    dpml = 2                # thickness of PML
    pml_layers = [mp.PML(dpml)]

    resolution = 100

    sr = b + pad + dpml            # radial size (cell is from 0 to sr)
    dimensions = mp.CYLINDRICAL    # coordinate system is (r,phi,z) instead of (x,y,z)
    cell = mp.Vector3(sr, 0, 0)

    m = 4

    geometry = [mp.Block(center=mp.Vector3(a + (w / 2)),
                         size=mp.Vector3(w, 1e20, 1e20),
                         material=mp.Medium(index=n))]

    # Finding a resonance mode with a high Q-value (calculated with Harminv)

    fcen = 0.15         # pulse center frequency
    df = 0.1            # pulse width (in frequency)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        resolution=resolution,
                        sources=sources,
                        dimensions=dimensions,
                        m=m)

    h = mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=200)

    resonance_0 = h.modes[0].freq

    sim.reset_meep()

    fcen = resonance_0
    df = 0.01

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        resolution=resolution,
                        sources=sources,
                        dimensions=dimensions,
                        m=m)

    sim.run(until_after_sources=200)

    npts = 10
    angles = 2 * np.pi / npts * np.arange(npts)
    parallel_fields = []
    for angle in angles:
        point = mp.Vector3(a, angle)
        e_z_field = sim.get_field_point(mp.Ez, point)
        e_total_field = np.real(np.sqrt(e_z_field * np.conj(e_z_field)))
        parallel_fields.append(e_total_field)

        point = mp.Vector3(b, angle)
        e_z_field = sim.get_field_point(mp.Ez, point)
        e_total_field = np.real(np.sqrt(e_z_field * np.conj(e_z_field)))
        parallel_fields.append(e_total_field)

    numerator_surface_integral = 2 * np.pi * b * mean(parallel_fields)
    denominator_surface_integral = sim.electric_energy_in_box(center=mp.Vector3((b + pad/2) / 2), size=mp.Vector3(b + pad/2))
    perturb_dw_dR = -resonance_0 * numerator_surface_integral / (4 * denominator_surface_integral)

    center_diff_dw_dR = []
    Harminv_resonances = []

    drs = np.logspace(start=-7, stop=-1.5, num=10)

    for dr in drs:
        sim.reset_meep()
        w = 1 + dr  # width of waveguide
        b = a + w

        fcen = resonance_0
        df = 0.01

        sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))]

        geometry = [mp.Block(center=mp.Vector3(a + (w / 2)),
                             size=mp.Vector3(w, 1e20, 1e20),
                             material=mp.Medium(index=n))]

        sim = mp.Simulation(cell_size=cell,
                            geometry=geometry,
                            boundary_layers=pml_layers,
                            resolution=resolution,
                            sources=sources,
                            dimensions=dimensions,
                            m=m)

        h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
        sim.run(mp.after_sources(h), until_after_sources=200)

        resonance_Harminv = h.modes[0].freq
        Harminv_resonances.append(resonance_Harminv)

        dw_dR = (resonance_dr - resonance_0) / dr
        center_diff_dw_dR.append(dw_dR)

    relative_errors_dw_dR = [abs((dw_dR - perturb_dw_dR) / perturb_dw_dR) for dw_dR in center_diff_dw_dR]

    predicted_resonances = [dr * perturb_dw_dR + resonance_0 for dr in drs]
    for i in len(Harminv_resonances):
        relative_errors_resonances = [abs((predicted_resonances[i] - Harminv_resonances[i]) / Harminv_resonances[i]) for resonance_dr in Harminv_resonances]

    if mp.am_master():
        plt.figure(dpi=150)
        plt.loglog(drs, relative_errors_dw_dR, 'bo-', label='relative error')
        plt.grid(True, which='both', ls='-')
        plt.xlabel('(perturbation amount $dr$)')
        plt.ylabel('relative error between \ncenter-difference and perturbation theory')
        plt.legend(loc='upper left')
        plt.title('Comparison of Perturbation Theory and \nCenter-Difference Calculations in Finding $dw/dR$')
        plt.tight_layout()
        #plt.show()
        plt.savefig('ring_Ez_perturbation_theory.dw_dR_error.png')
        plt.clf()

        plt.figure(dpi=150)
        plt.loglog(drs, relative_errors_resonances, 'bo-', label='relative error')
        plt.grid(True, which='both', ls='-')
        plt.xlabel('(perturbation amount $dr$)')
        plt.ylabel('relative error between resonances predicted resonances and resonances found with Harminv')
        plt.legend(loc='upper left')
        plt.title('Comparison of resonances predicted by $dw/dR$ found \nwith perturbation theory and resonances found \nwith Harminv in a separate simulation of perturbed state')
        plt.tight_layout()
        # plt.show()
        plt.savefig('ring_Ez_perturbation_theory.resonances_error.png')
        plt.clf()


if __name__ == '__main__':
    main()