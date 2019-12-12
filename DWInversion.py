import numpy as np
import numpy.ma as ma
# import l3msgen.functions.qaa as qaa
import pandas as pd
import os


def nechad(rho, a, c):
    ssc = a * rho / (1 - (rho / c))
    return ssc


def get_coefs(algoritm, rowname, sat=None):
    """ Gets coefs from txt file in auxdata for algos

            :param algo: <> name in nechad_2010, nechad_2016, han, dogliotti
            :param rowname: row name in first column file <algo>_<sat>.txt
            :param sat: as in .txt file algo_<sat>.txt
            :return: a_rho and c_rho from coefs files in auxdata
            """
    root = os.path.dirname(os.path.realpath(__file__))
    if sat:
        suffix_file = '_' + sat
    else:
        suffix_file = ''
    file = os.path.normpath(os.path.join(root, 'auxdata', algoritm + suffix_file + '.txt'))
    print(file)
    coefs = pd.read_csv(file, sep="\t", index_col=0)
    #print(algoritm," ",sat)
    return coefs.loc[rowname]['A_rho'], coefs.loc[rowname]['C_rho']


class DWInversionAlgos:
    def __init__(self):
        self.m = [0.005, 4.26, 0.52, 10.8]
        self.gamma = 0.265
        self.nodata = np.nan

    def Kd_lee(self, a, bb, bbw, sza):
        """ Computes diffuse light attenuation Kd

        for band following lee et al. 2016
        :param a: total absorption coefficient in m-1
        :param bb: total backscattering coefficient in m-1
        :param bbw: water backscattering coefficient in m-1
        :param sza: solar zenith angle in deg.
        :return: kd
        """
        m = self.m
        g = self.gamma
        Kd = (1. + m[0] * sza) * a + (1 - g * bbw / bb) * m[1] * (1 - m[2] * np.exp(-m[3] * a)) * bb
        return Kd

    # TODO delete ZSD in Input
    def ZSD_lee(self, Kd, Rrs, ZSD):
        """Computes Zecchi disk depth

        from Lee et al, RSE 2015
        :param Kd: spectral diffuse attenuation coef in m-1
        :param Rrs: above-water Rrs in sr-1
        :return: ZSD: Secchi depth in m
        """
        #Crt: threshold contrast for white disk in sr-1
        Crt = 0.013

        for i in range(Kd.shape[1]):
            try:
                idx = np.nanargmin(Kd[:, i])
                u = np.abs(0.14 - Rrs[idx, i]) / Crt
                ZSD[i] = 1. / (2.5 * Kd[idx, i]) * np.log(u)
            except:
                continue
        return

    def aCDOM_brezonik(self, b1_old, b2_old, sat='S2'):
        """Computes CDOM absorption coefficient (@440 nm).

        after Brezonik et al., 2015.
        Args:
            b1: blue or green band (surface reflectance)
            b2: red band (surface reflectance)
            sat: satellite, e.g. S2, S3 or L8

        Returns:
            CDOM absorption coefficient (@440 nm) band.
        """

        b1 = np.where((b1_old > 0) & (b2_old > 0), b1_old, b1_old+0.005)
        b2 = np.where((b1_old > 0) & (b2_old > 0), b2_old, b2_old+0.005)


        np.warnings.filterwarnings('ignore')
        if 'S2' in sat:
            a1, a2, = 1.872, -0.830
        elif 'S3' in sat:
            a1, a2 = 2.038, -0.832
        elif 'L8' in sat:
            a1, a2 = 1.582, -1.507
        acdom = np.where(np.logical_and(b1 >= 0, b2 >= 0), np.exp(a1 + a2 * np.log(b1 / b2)), self.nodata)
        return np.where(acdom >= 0, acdom, self.nodata)

    def aCDOM_perso(self, b1, b2):
        """Computes CDOM absorption coefficient (@440 nm).

        Args:
            b1: green band (surface reflectance)
            b2: red band (surface reflectance)

        Returns:
            CDOM absorption coefficient (@440 nm) band.
        """
        np.warnings.filterwarnings('ignore')
        a1, a2, = 0.01821993, 0.03822846
        acdom = a1 * (b1 / b2) + a2
        acdom = np.where(np.logical_and(b1 >= 0, b2 >= 0, acdom >= 0), acdom, self.nodata)
        return np.where(acdom >= 0, acdom, self.nodata)

# TODO check Alacantara
    def aCDOM_alcantara(self, b1, b2):
        """Computes CDOM absorption coefficient (@440 nm).

        after Alcantara et al., 2016.
        Args:
            b1: red band (surface reflectance)
            b2: blue band (surface reflectance)

        Returns:
            CDOM absorption coefficient (@440 nm) band.
        """
        np.warnings.filterwarnings('ignore')
        a1, a2, a3 = 2.7, -6.14, 4.19
        acdom = a1 * (b1 / b2)**2 + a2 * (b1 / b2) + a3
        acdom = np.where(np.logical_and(b1 >= 0, b2 >= 0, acdom >= 0), acdom, self.nodata)
        return np.where(acdom >= 0, acdom, self.nodata)

    def chl_gons(self, rho_red, rho_rededge, rho_nir, aw665, aw705, aphy_star=0.01):
        #TODO path awlanda to optional argument
        """Computes Chlorophyll concentration from 3 red bands after Gons et al., 99

        from Gons et al. 1999, 2004
        :param rho_red: rho_665 B4
        :param rho_rededge: rho_705 B5
        :param rho_nir: rho_775 or 740 B7 or B6
        :param aw665: water absorption at 665 nm
        :param aw705: water absorption at 705 nm
        :param aphy_star: Mean specific-chl absorption coefficient m2 mg-1 at 665 nm
        :return: chl mg.m-3
        """
        # from Gons et al. 1999, 2002
        np.warnings.filterwarnings('ignore')
        p = 1.06
        bb775 = (1.61 * rho_nir) / (0.082 - 0.6 * rho_nir)
        aphy = rho_rededge / rho_red * (aw705 + bb775) - aw665 - np.power(bb775, p)
        chl = aphy / aphy_star
        return chl

    def chl_lins(self, rho_red, rho_rededge):
        #TODO gerer homogeneisation resolution en entrée (20m) et autres coefs
        """Compute Chlorophyll concentration for turbid waters from red and rededge reflectances

        After Lins et al., 2017
        :param rho_red: Reflectances 665 nm MSI
        :param rho_rededge: Reflectances 705 nm MSI
        :return:  chl mg.m-3
        """
        def chlorolins(red, rededge):
            chl = 39.07 * (rededge / red) - 23.40
            return chl
        np.warnings.filterwarnings('ignore')
        chl_conc = np.where(np.logical_and(rho_red > 0, rho_rededge > 0),
                            chlorolins(rho_red, rho_rededge),
                            self.nodata)
        chl_conc = np.where(chl_conc < 0, self.nodata, chl_conc)
        return chl_conc

    # TODO set discrete indexes from Mishra 2012
    # def NDCI(self, rho_red, rho_rededge):
    #     """Compute Normalised Chlorophyll Index from red and rededge reflectances
    #
    #     After Mishra et al., 2012
    #     :param rho_red: Surface Reflectances 665 nm MSI
    #     :param rho_rededge: Surface Reflectances 705 nm MSI
    #     :return:  chl mg.m-3
    #     """
    #
    #     def ndci(red, rededge):
    #         index = (rededge - red) / ( rededge + red)
    #         return index
    #     np.warnings.filterwarnings('ignore')
    #     NDCI = np.where(np.logical_and(rho_red > 0, rho_rededge > 0),
    #                         ndci(rho_red, rho_rededge),
    #                         self.nodata)
    #     NDCI = np.where(NDCI < 0, self.nodata, NDCI)
    #
    #     return chl_conc
    #

    def qaa(self, iw, Rrs, N, wl, idx, aw, bbw, idwl):
        """Computes IOP from Rrs following QAA V6

        From lee et al. 2002, 2005
        :param width:
        :param Rrs: Rrs of iband
        :param N:
        :param wl:
        :param idx:
        :param aw:
        :param bbw:
        :param idwl:
        :return:
        """
        # print(N, wl, idx, aw[idwl], bbw[idwl], Rrs[:, iw])
        return qaa.qaa(N, wl, idx, aw[idwl], bbw[idwl], Rrs[:, iw])

    def ftest(self, a, b, c):
        print(a, b, c)

    def ndvi(self, R, NIR):
        return (NIR - R) / (NIR + R)

    def ndwi(self, rho_green, rho_nir):
        """Modification of Normalised Difference Water Index from green and swir band

        from McFeeters, 1996
        :param rho_green: Reflectances B3 @ 560 for S2 & @ 560 L8 dimentionless]
        :param rho_nir: Reflectances B8 @ 842 for S2, B5 @ 865 for L8 [dimentionless]
        :return: NDWI
        """
        ndwi = (rho_green - rho_nir) / (rho_green + rho_nir)
        return ndwi

    def mndwi(self, rho_green, rho_swir):
        """Modification of normalised difference water index from green and swir band

        from Xu, 2006
        :param rho_green: Reflectances B3 @ 560 for S2 & @ 560 L8 [dimentionless]
        :param rho_swir: Reflectances B11 @ 1640 for S2, B6 @ 1608 for L8 [dimentionless]
        :return: MNDWI
        """
        mndwi = (rho_green - rho_swir) / (rho_green + rho_swir)
        return mndwi

    # TODO include MuWI multiple band water index Wang et al., 2018

    def SPM_Nechad(self, rho, band='B4', sat="S2A", year='2010'):
        """Semi-analytical algorithm computes concentration SPM in mg/l from surface reflectances

        Nechad et al. 2010 and 2016 interpolated values with RSR same as in acolite (RBINS, VanHellemont)
        :param rho: Reflectances [dimentionless]
        :param band: satellite band by code
        :param sat: satellite type
        :param year: interpolated Nechad et al., 2010 or 2016
        :return: spm in mg.l-1
        """

        if 'S2' in sat:
            sat = 'S2'
        a_rho, c_rho = get_coefs(str('nechad_'+year),sat=sat,rowname=band)
        print('Nechad year {0} ; coefs Arho: {1}, Crho: {2}'.format(year,a_rho,c_rho))

        spm = nechad(rho, a_rho, c_rho)
        np.warnings.filterwarnings('ignore')
        spm = np.where(rho < 0, self.nodata, spm)
        return np.where(spm < 0, self.nodata, spm)

    def SPM_GET(self, rho_red, rho_nir, sat='S2A'):
        """ Switching hybrid algorithm computes concentration SPM in mg/L from red and NIR reflectances

        from SPM Nechad following Nechad et al. 2010 and interpolated values 2016 and RECOVER-GET band ratio
        :param water_mask: water mask
        :param rho_red: surface Reflectances red band [dl]
        :param rho_nir: surface Reflectances nir band [dl]
        :param sat: satellite type
        :return: spm in mg.l-1
        """
        band, year = 'B4', '2016'
        if 'S2' in sat:
            sat = 'S2'
        elif 'LANDSAT' in sat:
            sat = 'L8'

        limit_inf, limit_sup = 0.1, 0.20
        a, c = get_coefs(str('nechad_' + year), sat=sat, rowname=band)

        np.warnings.filterwarnings('ignore')
        red_pos = np.where(rho_red > 0, rho_red, self.nodata)
        nir_pos = np.where(rho_nir > 0, rho_nir, self.nodata)

        tsm_low = nechad(red_pos, a, c)
        tsm_high = 691.13 * np.power((nir_pos / red_pos), 2.5411)
        w = (rho_red - limit_inf) / (limit_sup - limit_inf)
        tsm_mixing = (1 - w) * tsm_low + w * tsm_high
        tsm = np.where(red_pos < limit_inf, tsm_low, self.nodata)
        tsm = np.where((red_pos >= limit_inf) & (red_pos <= limit_sup), tsm_mixing , tsm)
        tsm = np.where(red_pos > limit_sup, tsm_high, tsm)

        tsm = np.where(tsm < 0, self.nodata, tsm)

        # if water_mask is not None:
        #     tsm = np.where(water_mask == 0, -9999, tsm)

        return tsm

    def SPM_Han(self, Rrs_red, sat='S2A'):
        """ Switching SAA computes SPM concentration in mg/L from red band

        SPM Han following Han et al. 2016
        :param Rrs_red: Remote Sensing Reflectances Rrs Red band [sr-1]
        :param sat: satellite type L8, S2A etc.
        :return: spm in mg.l-1
        """

        if 'S2' in sat:
            sat = 'S2'
        if 'L8' in sat:
            limit_inf, limit_sup = 0.03, 0.045
        else:
            limit_inf, limit_sup = 0.03, 0.04
        a_low, c_low = get_coefs('han', sat=sat, rowname='SAAL')
        a_high, c_high = get_coefs('han', sat=sat, rowname='SAAHR')

        np.warnings.filterwarnings('ignore')
        Rrs_f = np.where(np.isnan(Rrs_red), self.nodata, Rrs_red)

        tsm_low = nechad(Rrs_f, a_low, c_low)
        tsm_high = nechad(Rrs_f, a_high, c_high)
        w_low = np.log10(limit_sup) - np.log10(Rrs_f)
        w_high = np.log10(Rrs_f) - np.log10(limit_inf)
        tsm_mixing = (w_low * tsm_low + w_high * tsm_high) / (w_low + w_high)
        tsm = np.where(Rrs_f < limit_inf, tsm_low, self.nodata)
        tsm = np.where((Rrs_f >= limit_inf) & (Rrs_f <= limit_sup), tsm_mixing, tsm)
        tsm = np.where(Rrs_f > limit_sup, tsm_high, tsm)
        return np.where(tsm < 0, self.nodata, tsm)

    def turb_Dogliotti(self, rho_red, rho_nir):
        """Switching semi-analytical-algorithm computes turbidity from red and NIR band

        following Dogliotti et al., 2015
        :param water_mask: mask with the water pixels (value=1)
        :param rho_red : surface Reflectances Red  band [dl]
        :param rho_nir: surface Reflectances  NIR band [dl]
        :return: turbidity in FNU
        """
        limit_inf, limit_sup = 0.05, 0.07
        a_low, c_low = get_coefs('dogliotti', sat='', rowname='Red')
        a_high, c_high = get_coefs('dogliotti', sat='', rowname='NIR')

        np.warnings.filterwarnings('ignore')
        #red_nonan = np.where(np.isnan(rho_red), self.nodata, rho_red)
        #nir_nonan = np.where(np.isnan(rho_nir), self.nodata, rho_nir)

        t_low = nechad(rho_red, a_low, c_low)
        t_high = nechad(rho_nir, a_high, c_high)
        w = (rho_red - limit_inf) / (limit_sup - limit_inf)
        t_mixing = (1 - w) * t_low + w * t_high
        turb = np.where(rho_red < limit_inf, t_low, self.nodata)
        turb = np.where((rho_red >= limit_inf) & (rho_red <= limit_sup), t_mixing, turb)
        turb = np.where(rho_red > limit_sup, t_high, turb)
        turb = np.where(turb < 0, self.nodata, turb)

        return turb

# factoriser et mettre r_converter ailleurs
def r_converter(M, ri, ro, F0=None):
    """ Reflectance converter function for one band

    :param M: input matrix
    :param ri: input reflectance type
    :param ro: output reflectance type
    :param F0: solar irradiance of band
    :return: converted matrix
    """
    if ri == 'Rrs' and ro == 'RHO':
        return M * np.pi
    elif ri == 'Lwn' and ro == 'RHO':
        return (M / (0.1 * F0)) * np.pi
    elif ri == 'RHO' and ro == 'Rrs':
        return M / np.pi
    elif ri == 'RHO' and ro == 'Lwn':
        return (M * (0.1 * F0)) / np.pi
