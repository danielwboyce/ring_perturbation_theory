from __future__ import division

import meep as mp
import numpy as np
from math import ceil
from statistics import mean
import matplotlib.pyplot as plt


def main():
    n = 3.4  # index of waveguide
    r = 1
    a = r  # inner radius of ring
    w = 1  # width of waveguide
    b = a + w  # outer radius of ring
    pad = 4  # padding between waveguide and edge of PML

    dpml = 2  # thickness of PML
    pml_layers = [mp.PML(dpml)]

    resolution = 100

    sr = b + pad + dpml  # radial size (cell is from 0 to sr)
    dimensions = mp.CYLINDRICAL  # coordinate system is (r,phi,z) instead of (x,y,z)
    cell = mp.Vector3(sr, 0, 0)

    m = 4

    geometry = [mp.Block(center=mp.Vector3(a + (w / 2)),
                         size=mp.Vector3(w, 1e20, 1e20),
                         material=mp.Medium(index=n))]

    # Finding a resonance mode with a high Q-value (calculated with Harminv)

    fcen = 0.15  # pulse center frequency
    df = 0.1  # pulse width (in frequency)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Hz, mp.Vector3(r + 0.1), amplitude=1)]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        resolution=resolution,
                        sources=sources,
                        dimensions=dimensions,
                        m=m)

    h = mp.Harminv(mp.Hz, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=200)

    print(f'Harminv found {len(h.modes)} resonant modes(s).')
    for mode in h.modes:
        print(f'The resonant mode with f={mode.freq} has Q={mode.Q}')

    #sim.plot2D(fields=mp.Ex,
    #       field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'},
    #       boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})
    #plt.show()


if __name__ == '__main__':
    main()