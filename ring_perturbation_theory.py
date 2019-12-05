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
    b = a+w                 # outer radius of ring
    pad = 4                 # padding between waveguide and edge of PML
    dpml = 2                # thickness of PML
    pml_layers = [mp.PML(thickness=dpml)]

    sxy = 2*(b+pad+dpml)  # cell size
    cell_size = mp.Vector3(sxy,sxy)
    resolution = 10

    c1 = mp.Cylinder(radius=b, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=a)
    geometry = [c1,c2]

    # Exciting the first resonance mode (calculated with Harminiv)

    fcen = 0.118101372125519    # pulse center frequency
    df = 0.1                    # pulse width (in frequency)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))]

    symmetries = [mp.Mirror(mp.Y)]

    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        symmetries=symmetries,
                        boundary_layers=pml_layers)

    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)),
            until_after_sources=300)

    # npts_inner = ceil(2 * np.pi * a / resolution)
    # npts_outer = ceil(2 * np.pi * b / resolution)
    npts_inner = 10
    npts_outer = 10
    angles_inner = 2 * np.pi / npts_inner * np.arange(npts_inner)
    angles_outer = 2 * np.pi / npts_outer * np.arange(npts_outer)

    inner_ring_fields = []
    outer_ring_fields = []
    for angle in angles_inner:
        point = mp.Vector3(a * np.cos(angle), a * np.sin(angle), 0)
        e_x_field = sim.get_field_point(mp.Ex, point)
        e_y_field = sim.get_field_point(mp.Ey, point)
        e_z_field = sim.get_field_point(mp.Ez, point)
        e_total_field = np.real(np.sqrt(e_x_field*np.conj(e_x_field) + e_y_field*np.conj(e_y_field) + e_z_field*np.conj(e_z_field)))
        #e_total_field = np.real(np.sqrt(e_z_field * np.conj(e_z_field)))
        inner_ring_fields.append(e_total_field)

    for angle in angles_outer:
        point = mp.Vector3(a * np.cos(angle), a * np.sin(angle), 0)
        e_x_field = sim.get_field_point(mp.Ex, point)
        e_y_field = sim.get_field_point(mp.Ey, point)
        e_z_field = sim.get_field_point(mp.Ez, point)
        e_total_field = np.real(np.sqrt(e_x_field*np.conj(e_x_field) + e_y_field*np.conj(e_y_field) + e_z_field*np.conj(e_z_field)))
        #e_total_field = np.real(np.sqrt(e_z_field * np.conj(e_z_field)))
        outer_ring_fields.append(e_total_field)

    surface_integral = 2 * np.pi * b * (mean(inner_ring_fields) + mean(outer_ring_fields)) / 2
    print(f'\nThe value of the surface_integral is {surface_integral}')


if __name__ == '__main__':
    main()