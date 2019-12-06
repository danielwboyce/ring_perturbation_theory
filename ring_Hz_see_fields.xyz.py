from __future__ import division

import meep as mp
import numpy as np
from math import ceil
from statistics import mean
import matplotlib.pyplot as plt


def main():
    n = 3.4  # index of waveguide
    w = 1  # width of waveguide
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 2  # thickness of PML
    sxy = 2 * (r + w + pad + dpml)  # cell size

    # Create a ring waveguide by two overlapping cylinders - later objects
    # take precedence over earlier objects, so we put the outer cylinder first.
    # and the inner (air) cylinder second.

    c1 = mp.Cylinder(radius=r + w, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r)

    # If we don't want to excite a specific mode symmetry, we can just
    # put a single point source at some arbitrary place, pointing in some
    # arbitrary direction.  We will only look for Ez-polarized modes.

    fcen = 0.15  # pulse center frequency
    df = 0.1  # pulse width (in frequency)

    src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Hz, mp.Vector3(r + 0.1))

    sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy),
                        geometry=[c1, c2],
                        sources=[src],
                        resolution=10,
                        symmetries=[mp.Mirror(mp.Y)],
                        boundary_layers=[mp.PML(dpml)])

    h = mp.Harminv(mp.Ex, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=200)

    print(f'Harminv found {len(h.modes)} resonant modes(s).')
    for mode in h.modes:
        print(f'The resonant mode with f={mode.freq} has Q={mode.Q}')

    sim.plot2D(fields=mp.Ex,
           field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'},
           boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})
    plt.show()


if __name__ == '__main__':
    main()