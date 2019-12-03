# Calculating 2d ring-resonator modes, from the Meep tutorial.
from __future__ import division

import meep as mp
import numpy as np

def main():
    n = 3.4                 # index of waveguide
    r = 1                   # radius of the cylinder
    a = r                   # inner radius of ring
    w = 1                   # width of waveguide
    b = a+w               # outer radius of ring
    pad = 4                 # padding between waveguide and edge of PML
    dpml = 2                # thickness of PML
    pml_layers = [mp.PML(thickness=dpml)]

    sxy = 2*(r+w+pad+dpml)  # cell size
    cell_size = mp.Vector3(sxy,sxy)
    resolution = 10

    # Create a ring waveguide by two overlapping cylinders - later objects
    # take precedence over earlier objects, so we put the outer cylinder first.
    # and the inner (air) cylinder second.

    c1 = mp.Cylinder(radius=b, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=a)
    geometry = [c1,c2]

    # If we don't want to excite a specific mode symmetry, we can just
    # put a single point source at some arbitrary place, pointing in some
    # arbitrary direction.  We will only look for Ez-polarized modes.

    fcen = 0.15             # pulse center frequency
    df = 0.1                # pulse width (in frequency)
    nfreq = 1               # number of frequencies to be tracked

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))]

    symmetries = [mp.Mirror(mp.Y)]

    npts_inner = 2*np.pi*a/resolution
    npts_outer = 2*np.pi*b/resolution
    angles_inner = 2*np.pi/npts_inner*np.arange(npts_inner)
    angles_outer = 2*np.pi/npts_outer*np.arange(npts_outer)

    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        symmetries=symmetries,
                        boundary_layers=pml_layers)

    inner_ring_points = []
    outer_ring_points = []
    for angle in angles:
        inner_ring_points.append(sim.add_flux(fcen=fcen,
                                              df=df,
                                              nfreq=nfreq,
                                              mp.FluxRegion(center=mp.Vector3(a*np.cos(angle),a*np.sin(angle)),size=mp.Vector3())))
        outer_ring_points.append(sim.add_flux(fcen=fcen,
                                              df=df,
                                              nfreq=nfreq,
                                              mp.FluxRegion(center=mp.Vector3(b*np.cos(angle), b*np.sin(angle)),
                                                            size=mp.Vector3())))


    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)),
            until_after_sources=300)

    # Output fields for one period at the end.  (If we output
    # at a single time, we might accidentally catch the Ez field when it is
    # almost zero and get a distorted view.)
    sim.run(mp.at_every(1/fcen/20, mp.output_efield_z), until=1/fcen)

if __name__ == '__main__':
    main()