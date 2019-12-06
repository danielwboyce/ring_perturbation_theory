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

    m = 5

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
    # print(f'\nThe value of numerator_surface_integral is {numerator_surface_integral}')

    denominator_surface_integral = sim.electric_energy_in_box(center=mp.Vector3((b + pad/2) / 2), size=mp.Vector3(b + pad/2))
    # print(f'\nThe value of denominator_surface_integral is {denominator_surface_integral}')

    perturb_dw_dR = -resonance_0 * numerator_surface_integral / (4 * denominator_surface_integral)
    print(f'\nThe value of perturb_dw_dR is {perturb_dw_dR}')

    drs = np.logspace(start=-7, stop=-1.5, num=10)
    print(f'My drs array is {drs}')
    center_diff_dw_dR = []
    for dr in drs:
        sim.reset_meep()
        w = 1 + dr  # width of waveguide
        print(f'My current dr is {dr}')
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

        resonance_dr = h.modes[0].freq
        dw_dR = (resonance_dr - resonance_0) / dr
        center_diff_dw_dR.append(dw_dR)
        print(f'When dr={dr}, dw_dR={dw_dR}')

    relative_errors = [abs((dw_dR - perturb_dw_dR) / perturb_dw_dR) for dw_dR in center_diff_dw_dR]
    if mp.am_master():
        plt.figure(dpi=150)
        plt.loglog(drs, relative_errors, 'bo-', label='relative error')
        plt.grid(True, which='both', ls='-')
        plt.xlabel('(perturbation amount $dr$)')
        plt.ylabel('relative error between \ncenter-difference and perturbation theory')
        plt.legend(loc='upper left')
        plt.title('Comparison of Perturbation Theory and \nCenter-Difference Calculations in Finding $dw/dR$')
        plt.tight_layout()
        #plt.show()
        plt.savefig('ring_Ez_perturbation_theory.png')


if __name__ == '__main__':
    main()