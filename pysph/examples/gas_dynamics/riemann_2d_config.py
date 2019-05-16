"""Different configurations for the 2D Riemann problems.

The different configurations and expected solutions to these problems
are defined in 'Solution of Two Dimensional Riemann Problems for Gas
Dynamics without Riemann Problem Solvers' by Alexander Kurganov and
Eitan Tadmor

General notations for the different configurations are (S) for shock
waves, (R) for Rarefactions and (J) for contact/slip lines

Code from https://bitbucket.org/kunalp/sph2d/
path: src/examples/r2d_config.py
"""


class R2DConfig(object):
    def __init__(self, config=3):
        self.config = config

        self.xmin = -0.25
        self.xmax = 1.15
        self.ymin = -0.25
        self.ymax = 1.15
        self.zmin = 0
        self.zmax = 0

        self.endtime = 0.25

        if config == 12:
            self.setup_config12()

        elif config == 2:
            self.setup_config2()

        elif config == 3:
            self.setup_config3()

        elif config == 4:
            self.setup_config4()

        elif config == 5:
            self.setup_config5()

        elif config == 6:
            self.setup_config6()

        elif config == 8:
            self.setup_config8()

        self.xmid = 0.5 * (self.xmin + self.xmax)
        self.ymid = 0.5 * (self.ymin + self.ymax)

        self.rho_max = max(self.rho1, self.rho2, self.rho3, self.rho4)
        self.rho_min = min(self.rho1, self.rho2, self.rho3, self.rho4)

    def setup_config3(self):
        """Four Shocks"""
        self.endtime = 0.3

        self.p1 = 1.5
        self.rho1 = 1.5
        self.u1 = 0.0
        self.v1 = 0.0

        self.p2 = 0.3
        self.rho2 = 0.5323
        self.u2 = 1.206
        self.v2 = 0.0

        self.p3 = 0.029
        self.rho3 = 0.138
        self.u3 = 1.206
        self.v3 = 1.206

        self.p4 = 0.3
        self.rho4 = 0.5323
        self.u4 = 0.0
        self.v4 = 1.206

    def setup_config2(self):
        """Four Rarefactions"""
        self.endtime = 0.2

        self.p1 = 1.0
        self.rho1 = 1.0
        self.u1 = 0.0
        self.v1 = 0.0

        self.p2 = 0.4
        self.rho2 = 0.5197
        self.u2 = -0.7259
        self.v2 = 0.0

        self.p3 = 1.0
        self.rho3 = 1.0
        self.u3 = -0.7259
        self.v3 = -0.7259

        self.p4 = 0.4
        self.rho4 = 0.5197
        self.u4 = 0.0
        self.v4 = -0.7259

    def setup_config4(self):
        self.endtime = 0.25

        self.p1 = 1.1
        self.rho1 = 1.1
        self.u1 = 0.0
        self.v1 = 0.0

        self.p2 = 0.35
        self.rho2 = 0.5065
        self.u2 = 0.8939
        self.v2 = 0.0

        self.p3 = 1.1
        self.rho3 = 1.1
        self.u3 = 0.8939
        self.v3 = 0.8939

        self.p4 = 0.35
        self.rho4 = 0.5065
        self.u4 = 0.0
        self.v4 = 0.8939

    def setup_config5(self):
        self.endtime = 0.23

        self.p1 = 1
        self.rho1 = 1
        self.u1 = -0.75
        self.v1 = -0.5

        self.p2 = 1.0
        self.rho2 = 2.0
        self.u2 = -0.75
        self.v2 = 0.5

        self.p3 = 1
        self.rho3 = 1
        self.u3 = 0.75
        self.v3 = 0.5

        self.p4 = 1.0
        self.rho4 = 3.0
        self.u4 = 0.75
        self.v4 = -0.5

    def setup_config6(self):
        self.endtime = 0.3

        self.p1 = 1
        self.rho1 = 1
        self.u1 = 0.75
        self.v1 = -0.5

        self.p2 = 1.0
        self.rho2 = 2.0
        self.u2 = 0.75
        self.v2 = 0.5

        self.p3 = 1
        self.rho3 = 1
        self.u3 = -0.75
        self.v3 = 0.5

        self.p4 = 1.0
        self.rho4 = 3.0
        self.u4 = -0.75
        self.v4 = -0.5

    def setup_config8(self):
        self.endtime = 0.25

        self.p1 = 0.4
        self.rho1 = 0.5197
        self.u1 = 0.1
        self.v1 = 0.1

        self.p2 = 1
        self.rho2 = 1.0
        self.u2 = -0.6259
        self.v2 = 0.1

        self.p3 = 1
        self.rho3 = 0.8
        self.u3 = 0.1
        self.v3 = 0.1

        self.p4 = 1.0
        self.rho4 = 1.0
        self.u4 = 0.1
        self.v4 = -0.6259

    def setup_config12(self):
        self.endtime = 0.25

        self.p1 = 0.4
        self.rho1 = 0.5313
        self.u1 = 0.0
        self.v1 = 0.0

        self.p2 = 1
        self.rho2 = 1.0
        self.u2 = 0.7276
        self.v2 = 0.0

        self.p3 = 1
        self.rho3 = 0.8
        self.u3 = 0.0
        self.v3 = 0.0

        self.p4 = 1.0
        self.rho4 = 1.0
        self.u4 = 0.0
        self.v4 = 0.7276


if __name__ == "__main__":
    config = R2DConfig()
