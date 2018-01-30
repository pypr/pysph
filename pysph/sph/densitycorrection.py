from pysph.sph.equation import Equation


class ShepardFilterPreStep(Equation):
    r"""**Shepard Filter density reinitialization**
    This is a zeroth order density reinitialization

    .. math::
            \tilde{W_{ab}} = \frac{W_{ab}}{\sum_{b} W_{ab}\frac{m_{b}}
            {\rho_{b}}}
    References
    ----------
    .. [Panizzo, 2004] Panizzo, Physical and Numerical Modelling of Subaerial
        Landslide Generated Waves, PhD thesis.

    """

    def initialize(self, d_idx, d_tw):
        d_tw[d_idx] = 0.0

    def loop(self, d_idx, d_tw, s_m, s_rho, s_idx, WIJ):
        d_tw[d_idx] += WIJ * s_m[s_idx] / s_rho[s_idx]


class ShepartFilter(Equation):
    r"""
    .. math::
            \rho_{a} = \sum_{b} \m_{b}\tilde{W_{ab}}
    """

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_tw, d_rho, WIJ):
        d_rho[d_idx] += WIJ * s_m[s_idx] / d_tw[d_idx]
