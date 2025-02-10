# Powerspectra from 
# https://github.com/DeaglanBartlett/symbolic_pofk/tree/main
# https://arxiv.org/abs/2311.15865
# https://arxiv.org/abs/2402.17492
# https://arxiv.org/abs/2410.14623


import numpy as np
#import scipy as scipy
#import warnings

def simpson(y, x):
    """
    Simplified and corrected implementation of Simpson's rule for equally spaced x values.
    Matches the behavior of scipy.integrate.simpson.
    """
    if len(y) != len(x):
        raise ValueError("y and x must have the same length.")
    if len(y) < 2:
        raise ValueError("At least 2 points are required for integration.")

    n = len(y) - 1  # Number of intervals
    h = (x[-1] - x[0]) / n  # Interval width (assumes equally spaced points)

    coefficients = np.ones_like(y)
    coefficients[0:-1:2] = 4  # Odd-indexed terms
    coefficients[1:-1:2] = 2  # Even-indexed terms

    integral = np.sum(coefficients * y) * h / 3
    return integral

def As_to_sigma8(As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated conversion As -> sigma8 

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """

    c = [0.0187, 2.4891, 12.9495, 0.7527,
         2.3685, 1.5062, 1.3057, 0.0885,
         0.1471, 3.4982, 0.006, 19.2779,
         11.1463, 1.5433, 7.0578, 2.0564]

    term1 = c[0] * (- Ob * c[1] + Om * c[2] +
                    np.log(- c[3] * w0 + np.log(- c[4] * w0 - c[5] * wa)))

    term2 = Om * c[6] + c[7] * mnu + c[8] * ns - \
        np.log(Om * c[9] - c[10] * wa)

    term3 = Ob * c[11] - Om * c[12] - ns

    term4 = - Om * c[13] - c[14] * h + c[15] * mnu + ns

    result = term1 * term2 * term3 * term4

    return result*np.sqrt(As)


def sigma8_to_As(sigma8, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated conversion sigma8 -> As

    Args:
        :sigma8 (float): The z=0 rms mass fluctuation in spheres of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :As (float): 10^9 times the amplitude of the primordial P(k)
    """

    c = [0.0187, 2.4891, 12.9495, 0.7527,
         2.3685, 1.5062, 1.3057, 0.0885,
         0.1471, 3.4982, 0.006, 19.2779,
         11.1463, 1.5433, 7.0578, 2.0564]

    term1 = c[0] * (- Ob * c[1] + Om * c[2] +
                    np.log(- c[3] * w0 + np.log(- c[4] * w0 - c[5] * wa)))

    term2 = Om * c[6] + c[7] * mnu + c[8] * ns - \
        np.log(Om * c[9] - c[10] * wa)

    term3 = Ob * c[11] - Om * c[12] - ns

    term4 = - Om * c[13] - c[14] * h + c[15] * mnu + ns

    result = term1 * term2 * term3 * term4

    return (sigma8/result)**2


def As_to_sigma8_max_precision(As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated conversion As -> sigma8, using the most accurate expression

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """

    b = np.array([0.0246, 2.1062, 2.9355, 0.7626, 0.2962, 0.5096, 
                  4.4025, 3.6495, 0.4144, 0.8615, 0.6188, 0.1751, 
                  0.824, 0.5466, 0.5519, 0.3689, 0.3261, 0.2002, 
                  0.8892, 0.4462, 1.215, 3.4829, 2.5852, 0.0242, 
                  0.0051, 0.1614, 1.2991, 4.1426, 3.3055, 0.5716, 
                  6.0094, 1.9569, 2.1477, 1.1902, 0.128, 0.6931, 
                  0.2661])

    term1_inner = (Om * b[1] + 
                   (b[2] * mnu - b[3] * ns + np.log(b[4] * h - b[5] * mnu)) *
                   (b[6] * h + b[7] * mnu - b[8] * ns + 1))
    term1 = b[0] * term1_inner
    
    term2 = b[9] * h - mnu

    term3_inner1 = (b[12] * w0 - b[13] * wa - np.log(Om * b[14])) * \
                   (Om * b[15] + b[16] * w0 + b[17] * wa + np.log(-b[18] * w0 - b[19] * wa))
    term3_inner2 = np.log(Om * b[20] + np.log(-b[21] * w0 - b[22] * wa))
    term3 = b[10] * w0 - b[11] * mnu - term3_inner1 - term3_inner2 + np.log(-b[23] * w0 - b[24] * wa)
    
    term4_inner1 = Ob * b[30] - b[31] * h - np.log(Om * b[32])
    term4_inner2 = Om * b[33] - b[34] * h - b[35] * mnu - b[36] * ns
    term4 = b[25] * mnu - np.sqrt(Ob) * b[26] - Ob * b[27] + Om * b[28] - b[29] * h + 1 + term4_inner1 * term4_inner2

    result = term1 * term2 * term3 * term4
    
    return result*np.sqrt(As)


def sigma8_to_As_max_precision(sigma8, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated conversion sigma8 -> As, using the most accurate expression

    Args:
        :sigma8 (float): The z=0 rms mass fluctuation in spheres of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :As (float): 10^9 times the amplitude of the primordial P(k)
    """

    b = np.array([0.0246, 2.1062, 2.9355, 0.7626, 0.2962, 0.5096, 
                  4.4025, 3.6495, 0.4144, 0.8615, 0.6188, 0.1751, 
                  0.824, 0.5466, 0.5519, 0.3689, 0.3261, 0.2002, 
                  0.8892, 0.4462, 1.215, 3.4829, 2.5852, 0.0242, 
                  0.0051, 0.1614, 1.2991, 4.1426, 3.3055, 0.5716, 
                  6.0094, 1.9569, 2.1477, 1.1902, 0.128, 0.6931, 
                  0.2661])

    term1_inner = (Om * b[1] + 
                   (b[2] * mnu - b[3] * ns + np.log(b[4] * h - b[5] * mnu)) *
                   (b[6] * h + b[7] * mnu - b[8] * ns + 1))
    term1 = b[0] * term1_inner
    
    term2 = b[9] * h - mnu

    term3_inner1 = (b[12] * w0 - b[13] * wa - np.log(Om * b[14])) * \
                   (Om * b[15] + b[16] * w0 + b[17] * wa + np.log(-b[18] * w0 - b[19] * wa))
    term3_inner2 = np.log(Om * b[20] + np.log(-b[21] * w0 - b[22] * wa))
    term3 = b[10] * w0 - b[11] * mnu - term3_inner1 - term3_inner2 + np.log(-b[23] * w0 - b[24] * wa)
    
    term4_inner1 = Ob * b[30] - b[31] * h - np.log(Om * b[32])
    term4_inner2 = Om * b[33] - b[34] * h - b[35] * mnu - b[36] * ns
    term4 = b[25] * mnu - np.sqrt(Ob) * b[26] - Ob * b[27] + Om * b[28] - b[29] * h + 1 + term4_inner1 * term4_inner2

    result = term1 * term2 * term3 * term4
    
    return (sigma8/result)**2


def growth_correction_R(As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Correction to the growth factor 

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): The scale factor to evaluate P(k) at

    Returns:
        :result (float): correction to the growth factor
    """

    d = np.array([0.8545, 0.394, 0.7294, 0.5347, 0.4662, 4.6669,
                  0.4136, 1.4769, 0.5959, 0.4553, 0.0799, 5.8311,
                  5.8014, 6.7085, 0.3445, 1.2498, 0.3756, 0.2136])

    part1 = d[0]

    denominator_inner1 = a * \
        d[1] + d[2] + (Om * d[3] - a * d[4]) * np.log(-d[5] * w0 - d[6] * wa)
    part2 = -1 / denominator_inner1

    numerator_inner2 = Om * d[7] - a * d[8] + np.log(-d[9] * w0 - d[10] * wa)
    denominator_inner2 = -a * d[11] + d[12] + d[13] * \
        (Om * d[14] + a * d[15] - 1) * (d[16] * w0 + d[17] * wa + 1)
    part3 = -numerator_inner2 / denominator_inner2

    result = 1 + (1 - a) * (part1 + part2 + part3)

    return result


def log10_S(k, As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Corrections to the present-day linear power spectrum

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :result (np.ndarray): Corrections to the present-day linear power spectrum
    """

    e = np.array([0.2841, 0.1679, 0.0534, 0.0024, 0.1183, 0.3971,
                  0.0985, 0.0009, 0.1258, 0.2476, 0.1841, 0.0316,
                  0.1385, 0.2825, 0.8098, 0.019, 0.1376, 0.3733])

    part1 = -e[0] * h
    part2 = -e[1] * w0
    part3 = -e[2] * mnu / np.sqrt(e[3] + k**2)

    part4 = -(e[4] * h) / (e[5] * h + mnu)

    part5 = e[6] * mnu / (h * np.sqrt(e[7] + (Om * e[8] + k)**2))

    numerator_inner = (e[9] * Ob - e[10] * w0 - e[11] * wa +
                       (e[12] * w0 + e[13]) / (e[14] * wa + w0))
    denominator_inner = np.sqrt(e[15] + (Om + e[16] * np.log(-e[17] * w0))**2)

    part6 = numerator_inner / denominator_inner

    # Sum all parts to get the final result
    result = part1 + part2 + part3 + part4 + part5 + part6

    return result/10


def get_approximate_D(k, As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Approximation to the growth factor using the results of
    Bond et al. 1980, Lahav et al. 1992, Carrol et al. 1992 
    and Eisenstein & Hu 1997 (D_cbnu).

    There are two differences between our method and theirs. 
    First, in Eisenstein & Hu 1997 D is chosen to be (1 + zeq) a at 
    early times, whereas we instead choose D -> a at early times. 
    Second, the formulae reported there assume that w=-1, whereas we
    change the Omega_Lambda terms to include a w0-wa parameterisation.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :D (np.ndarray): Approximate linear growth factor at corresponding k values
    """

    # avoid singularities
    mnu = mnu + 1e-10

    #  Get fitting formula without free-streaming
    z = 1 / a - 1
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin
    zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4

    Omega = Om * a ** (-3)
    OL = (1 - Om) * a ** (-3 * (1 + w0 + wa)) * np.exp(- 3 * wa * (1 - a))
    g = np.sqrt(Omega + OL)
    Omega /= g ** 2
    OL /= g ** 2

    D1 = (
        (1 + zeq) / (1 + z) * 5 * Omega / 2 /
        (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
    )

    # Split Omega_m into CDM, Baryons and Neutrinos
    Onu = mnu / 93.14 / h ** 2
    Oc = Om - Ob - Onu
    fc = Oc / Om
    fb = Ob / Om
    fnu = Onu / Om
    fcb = fc + fb

    # Add Bond et al. 1980 suppression
    pcb = 1/4 * (5 - np.sqrt(1 + 24 * fcb))
    Nnu = (3 if mnu != 0.0 else 0)
    q = k * h * theta2p7 ** 2 / (Om * h ** 2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7/6)) * (Nnu * q / fnu) ** 2
    Dcbnu = (fcb ** (0.7/pcb) + (D1 / (1 + yfs)) **
             0.7) ** (pcb / 0.7) * D1 ** (1 - pcb)

    # Remove 1+zeq normalisation given in Eisenstein & Hu 1997
    D = Dcbnu / (1 + zeq)

    return D


def get_eisensteinhu_nw(k, As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the no-wiggles Eisenstein & Hu approximation
    to the linear P(k) at redshift zero.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum

    Returns:
        :pk (np.ndarray): Approxmate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    ombom0 = Ob / Om
    om0h2 = Om * h**2
    ombh2 = Ob * h**2
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

    # Compute scale factor s, alphaGamma, and effective shape Gamma
    s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * \
        np.log(431.0 * om0h2) * ombom0 + 0.38 * \
        np.log(22.3 * om0h2) * ombom0**2
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                      (1.0 + (0.43 * k * h * s)**4))

    # Compute q, C0, L0, and tk_eh
    q = k * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    kpivot = 0.05

    pk = (
        2 * np.pi ** 2 / k ** 3
        * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
        * (2 * k ** 2 * 2998**2 / 5 / Om) ** 2
        * tk_eh ** 2
    )

    return pk

def pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns, integral_norm=True):
    """
    Compute the Eisentein & Hu 1998 zero-baryon approximation to P(k) at z=0

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :use_colossus (bool, default=False): Whether to use the external package colossus
            to compute this term
        :integral_norm (bool, default=True): Whether to compute the normalisation of the
            power spectrum using an integral over k

    Returns:
        :pk_eh (np.ndarray): The Eisenstein & Hu 1998 zero-baryon P(k) [(Mpc/h)^3]
    """

    if integral_norm:
        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

        def get_pk(kk, Anorm):

            # Compute scale factor s, alphaGamma, and effective shape Gamma
            s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
            alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0h2) * ombom0 + \
                0.38 * np.log(22.3 * om0h2) * ombom0**2
            Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                              (1.0 + (0.43 * kk * h * s)**4))

            # Compute q, C0, L0, and tk_eh
            q = kk * theta2p7**2 / Gamma
            C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
            L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
            tk_eh = L0 / (L0 + C0 * q**2)

            # Calculate Pk with unit amplitude
            return Anorm * tk_eh**2 * kk**ns

        # Define integration bounds and number of sub-intervals
        b0 = np.log(1e-7)  # ln(k_min)
        b1 = np.log(1e5)  # ln(k_max)
        # Number of sub-intervals (make sure it's even for Simpson's Rule)
        n = 1000

        # Find normalisation
        R = 8.0
        kk = np.exp(np.linspace(b0, b1, n))
        x = kk * R
        W = np.zeros(x.shape)
        m = x < 1.e-3
        W[m] = 1.0
        W[~m] = 3.0 / x[~m]**3 * (np.sin(x[~m]) - x[~m] * np.cos(x[~m]))
        y = get_pk(kk, 1.0) * W**2 * kk**3
        sigma2 = simpson(y, x=np.log(x))

        sigmaExact = np.sqrt(sigma2 / (2.0 * np.pi**2))
        Anorm = (sigma8 / sigmaExact)**2

        pk_eh = get_pk(k, Anorm)
    else:
        As = sigma8_to_As(sigma8, Om, Ob, h, ns)

        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

        # Compute scale factor s, alphaGamma, and effective shape Gamma
        s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
        alphaGamma = 1.0 - 0.328 * \
            np.log(431.0 * om0h2) * ombom0 + 0.38 * \
            np.log(22.3 * om0h2) * ombom0**2
        Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                          (1.0 + (0.43 * k * h * s)**4))

        # Compute q, C0, L0, and tk_eh
        q = k * theta2p7**2 / Gamma
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
        tk_eh = L0 / (L0 + C0 * q**2)

        kpivot = 0.05

        pk_eh = (
            2 * np.pi ** 2 / k ** 3
            * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
            * (2 * k ** 2 * 2998**2 / 5 / Om) ** 2
            * tk_eh ** 2
        )

        #  Get fitting formula without free-streaming
        a = 1.0
        z = 1 / a - 1
        theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin
        zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4

        Omega = Om * a ** (-3)
        OL = (1 - Om)
        g = np.sqrt(Omega + OL)
        Omega /= g ** 2
        OL /= g ** 2

        D1 = (
            (1 + zeq) / (1 + z) * 5 * Omega / 2 /
            (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
        )
        D1 /= (1 + zeq)
        pk_eh *= D1 ** 2

    return pk_eh

def lcdm_logF_fiducial(k, sigma8, Om, Ob, h, ns, extrapolate=False, kmin=9.e-3, kmax=9):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the fiducial exprssion
    given in Bartlett et al. 2023.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for k < kmin and k > kmax. Due to problems with the Colossus
            Eisenstein & Hu fit on large scales (typically a 2 percent offset), it is
            strongly recommended to use extrapolate=False.
        :kmin (float, default=9.e-3): Minimum k value to use Bartlett et al. formula
            if extrapolate=False
        :kmax (float, default=9): Maximum k value to use Bartlett et al. formula
            if extrapolate=False

    Returns:
        :logF (np.ndarray): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """

    b = [0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
         4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
         0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
         0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
         70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
         4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
         0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
         7.385, 12.3960625361899, 0.0134114370723638]

    line1 = b[0] * h - b[1]

    line2 = (
        ((Ob * b[2]) / np.sqrt(h ** 2 + b[3])) ** (b[4] * Om) *
        (
            (b[5] * k - Ob) / np.sqrt(b[6] + (Ob - b[7] * k) ** 2)
            * b[8] * (b[9] * k) ** (-b[10] * k) * np.cos(Om * b[11]
                                                         - (b[12] * k) / np.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k) / np.sqrt(1 + b[16] * k ** 2) - Om)
            * np.cos(b[17] * h / np.sqrt(1 + b[18] * k ** 2))
        )
    )

    line3 = (
        b[19] * (b[20] * Om + b[21] * h - np.log(b[22] * k)
                 + (b[23] * k) ** (- b[24] * k)) * np.cos(b[25] / np.sqrt(1 + b[26] * k ** 2))
    )

    line4 = (
        (b[27] * k) ** (-b[28] * k) * (b[29] * k - (b[30] * np.log(b[31] * k))
                                       / np.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * np.cos(Om * b[34] - (b[35] * k) / np.sqrt(Ob ** 2 + b[36]))
    )

    logF = line1 + line2 + line3 + line4

    # Use Bartlett et al. 2023 P(k) only in tested regime
    m = ~((k >= kmin) & (k <= kmax))
    if (not extrapolate) and np.sum(m) > 0:
        print(
            "Not using Bartlett et al. formula outside tested regime")
        logF[m] = np.log(
            pk_EisensteinHu_zb(k[m], sigma8, Om, Ob, h, ns, integral_norm=False) /
            pk_EisensteinHu_zb(k[m], sigma8, Om, Ob, h, ns, integral_norm=True)
        )
    elif np.sum(m) > 0:
        print(
            "Using Bartlett et al. formula outside tested regime. Due to problems with the Colossus Eisenstein & Hu fit on \
                large scales (typically a 2 percent offset), it is strongly recommended to use extrapolate=False.")

    return logF


def logF_fiducial(k, As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated logarithm of the ratio between the true linear power spectrum 
    and the Eisenstein & Hu 1998 fit for LCDM as given in Bartlett et al. 2023.

    This calls the logF_fiducial function from symbolic_pofk.linear but extrapolates
    to all k considered

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :logF (np.ndarray): The emulated logarithm of the ratio between the true linear power spectrum
    """

    sigma8 = None  # not needed in logF_fiducial
#    with warnings.catch_warnings():
#       warnings.filterwarnings("ignore")
    logF = lcdm_logF_fiducial(k, sigma8, Om, Ob, h, ns, extrapolate=True)

    return logF


def plin_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a=1):
    """
    Compute the emulated linear matter power spectrum by combining the Eisenstein & Hu model, an approximation for the growth factor D, 
    the fit from Bartlett et al. (2023), and corrections to both the present-day linear power spectrum and the growth factor.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float, default=1): The scale factor to evaluate P(k) at

    Returns:
        :pk_lin (np.ndarray): The emulated linear P(k) [(Mpc/h)^3]
    """

    eh = get_eisensteinhu_nw(k, As, Om, Ob, h, ns, mnu, w0, wa)
    D_value = get_approximate_D(k, As, Om, Ob, h, ns, mnu, w0, wa, a)
    logF_value = logF_fiducial(k, As, Om, Ob, h, ns, mnu, w0, wa)

    F_value = np.exp(logF_value)
    R_value = growth_correction_R(As, Om, Ob, h, ns, mnu, w0, wa, a)
    log10_S_value = log10_S(k, As, Om, Ob, h, ns, mnu, w0, wa)
    S_value = np.power(10, log10_S_value)

    pk_lin = eh * D_value ** 2 * F_value * R_value * S_value

    return pk_lin

def pnl_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Compute the non-linear power spectrum using a symbolic approximation
    to the linear power spectrum. 

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :pk_nl (np.ndarray): The emulated non-linear P(k) [(Mpc/h)^3]
    """

    g = np.array([0.2107, 0.0035, 0.0667, 0.0442, 1.2809, 0.2287, 0.1122, 4.3318, 1.1857, 3.3117, 14.2829,
                 0.9039, 0.0749, 0.0741, 0.1277, 27.6818, 24.8736, 0.6264, 0.3035, 0.6069, 0.7882, 0.4811,
                 1.4326, 1.8971, 0.0271, 0.9635, 0.0264, 22.9213, 71.1658, 0.0371, 0.0099, 210.3925, 0.2555])

    # calculate the linear power spectrum using the emulated model
    P_lin = np.log10(logF_fiducial_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a))

    term1 = P_lin

    numerator1 = g[0] * k * (g[1] * k)**(g[2] * Om - g[3] * As)
    denominator1_part1 = (g[4] * k**(-g[5]) - g[6] *
                          P_lin)**(g[7] * P_lin + g[8] * wa + g[9] * w0 - g[10])
    denominator1_part2 = (g[11] * k**g[12] + g[13] *
                          P_lin - g[14] * Om)**(g[15] * a - g[16] * ns)
    term2 = numerator1 / (denominator1_part1 + denominator1_part2)

    numerator2 = (g[17] * a - g[18] * P_lin + g[19]) * k
    denominator2 = (g[20] * Om + g[21] * k + g[22] * ns - g[23] +
                    (g[24] * P_lin + g[25] * k**g[26])**(g[27] * a - g[28] * ns))
    term3 = numerator2 / denominator2

    term4 = g[29] * k

    term5 = (g[30] * k)**((g[31] * k)**(-a * g[32]))

    # Combine all terms
    pk_nl = term1 + term2 + term3 - term4 - term5

    bias = pnl_bias(k)
    pk_nl = pk_nl - bias

    return np.power(10, pk_nl)


def pnl_bias(k):
    """
    the offset of the emulated non-linear power spectrum from euclidemulator2

    Parameters:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]

    Returns:
        :offset (np.ndarray): The offset of the emulated non-linear power spectrum
    """

    h = np.array([0.5787,   2.3485,  27.3829,  16.4236,  97.3766,  90.9764,
                  11.2046,  2447.2, 11376.93])

    term1 = ((h[1] * k) - np.cos(h[3]*np.cos(h[2] * k))) * np.cos(h[4] * k)
    denominator = -h[7]*np.log(h[6] * k) + (h[8] * k)

    offset = ((h[0] + term1 + np.cos(h[5] * k))) / denominator

    return offset

def test_linear():
    # Define k range
    kmin = 1e-4
    kmax = 90
    nk = 900
    extrapolate_kmin = 2e-3
    extrapolate_kmax = kmax
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
   
    # Cosmological parameters
    As = 2.105  # 10^9 A_s
    h = 0.6766
    Om = 0.3111
    Ob = 0.02242 / h ** 2
    ns = 0.9665
#    tau = 0.0561
    mnu = 0.10
    w0 = -1.0
    wa = 0.

# Redshift
    z = 0
    a = 1 / (1+z)
    
    # Get sigma8 for this As
    sigma8 = As_to_sigma8(As, Om, Ob, h, ns, mnu, w0, wa)
    print('sigma8 = ', sigma8)

    # See what As you get in reverse
    As_new = sigma8_to_As(sigma8, Om, Ob, h, ns, mnu, w0, wa)
    print('As_new = ', As_new)

    # Emulate P(k)
    pk_eh = pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns)
#    pk_eh_b = pk_EisensteinHu_b(k, sigma8, Om, Ob, h, ns)
    pk_fid = plin_new_emulated(
        k, As, Om, Ob, h, ns, mnu, w0, wa, a=a)
    pk_nl = pnl_new_emulated(
        k, As, Om, Ob, h, ns, mnu, w0, wa, a=a)
    pk_prec = pk_fid
#    logF_eh_b = np.log(pk_eh_b / pk_eh)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    cmap = plt.get_cmap('Set1')
    axs.loglog(k, pk_nl,
                    label='Non-linear matter power spectrum', color=cmap(1),alpha=0.37,lw=1.5)
    axs.loglog(k, pk_eh,
                    label='Zero Baryon (Eisenstein & Hu 1998)', color=cmap(0),alpha=0.37,lw=3)
    axs.loglog(k, pk_fid,
                    label='Fiducial (Bartlett et al. 2023)', color=cmap(2),alpha=0.97,lw=1.5)
#    axs[0].semilogx(k, pk_camb / pk_camb, label='camb', color=cmap(4), ls='--')
#    axs[1].semilogx(k, logF_eh_b, label='Baryon', color=cmap(1))
#    axs[1].semilogx(k, logF_camb, label='camb', color=cmap(4), ls='--')
    axs.legend()
    axs.set_xlabel(r'$k \ / \ h {\rm \, Mpc}^{-1}$')
    axs.set_ylabel(r'$P(k) / P_{\rm prec}(k)$')
    axs.set_ylabel(r'$\log F$')
    axs.axhline(0, color='k')
    fig.align_labels()
    fig.tight_layout()
    #fig.savefig('planck_2018_comparison.png', bbox_inches='tight')

    
    plt.show()
    
# test_linear()