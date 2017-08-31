from argparse import ArgumentParser

from pysph.sph.scheme import SchemeChooser, WCSPHScheme
from pysph.sph.wc.edac import EDACScheme


def test_scheme_chooser_does_not_clobber_default():

    # When
    wcsph = WCSPHScheme(
        ['f'], ['b'], dim=2, rho0=1.0, c0=10.0,
        h0=0.1, hdx=1.3, alpha=0.2, beta=0.1,
    )
    edac = EDACScheme(
        fluids=['f'], solids=['b'], dim=2, c0=10.0, nu=0.001,
        rho0=1.0, h=0.1, alpha=0.0, pb=0.0
    )
    s = SchemeChooser(default='wcsph', wcsph=wcsph, edac=edac)
    p = ArgumentParser(conflict_handler="resolve")
    s.add_user_options(p)
    opts = p.parse_args([])

    # When
    s.consume_user_options(opts)

    # Then
    assert s.scheme.alpha == 0.2
    assert s.scheme.beta == 0.1

    # When
    opts = p.parse_args(['--alpha', '0.3', '--beta', '0.4'])
    s.consume_user_options(opts)

    # Then
    assert s.scheme.alpha == 0.3
    assert s.scheme.beta == 0.4
