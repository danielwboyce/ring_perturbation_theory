from __future__ import division

import meep as mp
import numpy as np
from math import ceil
from statistics import mean


def main():
    n = 3.4                 # index of waveguide
    r = 1
    a = r                   # inner radius of ring
    w = 1                   # width of waveguide
    b = a + w               # outer radius of ring
    pad = 4                 # padding between waveguide and edge of PML

    dpml = 2                # thickness of PML
    pml_layers = [mp.PML(dpml)]

    resolution = 10

    sr = b + pad + dpml            # radial size (cell is from 0 to sr)
    dimensions = mp.CYLINDRICAL    # coordinate system is (r,phi,z) instead of (x,y,z)
    cell = mp.Vector3(sr, 0, 0)

    m = 1

    geometry = [mp.Block(center=mp.Vector3(a + (w / 2)),
                         size=mp.Vector3(w, 1e20, 1e20),
                         material=mp.Medium(index=n))]

    # Finding a resonance mode with a high Q-value (calculated with Harminv)

    fcen = 0.5         # pulse center frequency
    df = 0.5            # pulse width (in frequency)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Er, mp.Vector3(r+0.1), amplitude=1),
               mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ep, mp.Vector3(r+0.1), amplitude=1j)]

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

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Er, mp.Vector3(r + 0.1), amplitude=1),
               mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ep, mp.Vector3(r + 0.1), amplitude=1j)]

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
    perpendicular_fields = []

    for angle in angles:
        point = mp.Vector3(a, angle)
        e_r_field = sim.get_field_point(mp.Er, point)
        temp_perpendicular_field = np.real(np.sqrt(e_r_field*np.conj(e_r_field)))
        perpendicular_fields.append(temp_perpendicular_field)

        e_p_field = sim.get_field_point(mp.Ep, point)
        e_z_field = sim.get_field_point(mp.Ez, point)
        temp_parallel_field = np.real(np.sqrt(e_p_field*np.conj(e_p_field) + e_z_field*np.conj(e_z_field)))
        parallel_fields.append(temp_parallel_field)

        point = mp.Vector3(b, angle)
        e_r_field = sim.get_field_point(mp.Er, point)
        temp_perpendicular_field = np.real(np.sqrt(e_r_field * np.conj(e_r_field)))
        perpendicular_fields.append(temp_perpendicular_field)

        e_p_field = sim.get_field_point(mp.Ep, point)
        e_z_field = sim.get_field_point(mp.Ez, point)
        temp_parallel_field = np.real(np.sqrt(e_p_field * np.conj(e_p_field) + e_z_field * np.conj(e_z_field)))
        parallel_fields.append(temp_parallel_field)

    numerator_surface_integral = 2 * np.pi * b * mean(parallel_fields)
    print(f'\nThe value of numerator_surface_integral is {numerator_surface_integral}')

    denominator_surface_integral = sim.electric_energy_in_box(center=mp.Vector3(), size=mp.Vector3(2 * (b + pad/2)))
    print(f'\nThe value of denominator_surface_integral is {denominator_surface_integral}')

    perturb_dw_dR = -resonance_0 * numerator_surface_integral / (4 * denominator_surface_integral)
    print(f'\nThe value of perturb_dw_dR is {perturb_dw_dR}')


if __name__ == '__main__':
    main()