import matplotlib.pyplot as plt
from pathlib import Path
from lxml import etree
import numpy as np


class DWGlintProcessor:
    supported_products = ['S2_S2COR', 'S2_THEIA', 'S2_PLANETARY']

    def __init__(self, image, limit_angle=30):

        self.image = image
        self.limit_angle = limit_angle
        self.adjustable_bands = ['Mir', 'Mir2', 'Nir', 'Nir2']

        try:
            self.glint_array = self.create_glint_array(self.image.granule_metadata, image.product)

        except BaseException as err:
            self.glint_array = None
            print(f'### GLINT PROCESSOR ERROR #####')
            print(err)

    @classmethod
    def create(cls, image, limit_angle=30):
        if image.product not in cls.supported_products:
            print(f'Product {image.product} not supported by GlintProcessor')
            print(f'Supported products: {cls.supported_products}')
            return None
        else:
            return cls(image, limit_angle)

    @staticmethod
    def get_grid_values_from_xml(tree_node, xpath_str):
        """Receives a XML tree node and a XPath parsing string and search for children matching the string.
           Then, extract the VALUES in <values> v1 v2 v3 </values> <values> v4 v5 v6 </values> format as numpy array
           Loop through the arrays to compute the mean.
        """
        node_list = tree_node.xpath(xpath_str)

        arrays_lst = []
        for node in node_list:
            values_lst = node.xpath('.//VALUES/text()')
            values_arr = np.array(list(map(lambda x: x.split(' '), values_lst))).astype('float')
            arrays_lst.append(values_arr)

        return np.nanmean(arrays_lst, axis=0)

    @staticmethod
    def create_glint_array(xml_file, product):
        xml_file = Path(xml_file)
        parser = etree.XMLParser()
        root = etree.parse(xml_file.as_posix(), parser).getroot()

        sun_angles = 'Sun_Angles_Grid' if product in ['S2_S2COR', 'S2_PLANETARY'] else 'Sun_Angles_Grids'
        # viewing_angles = 'Viewing_Incidence_Angles_Grids'

        sun_zenith = np.deg2rad(DWGlintProcessor.get_grid_values_from_xml(root, f'.//{sun_angles}/Zenith'))[:-1, :-1]
        sun_azimuth = np.deg2rad(DWGlintProcessor.get_grid_values_from_xml(root, f'.//{sun_angles}/Azimuth'))[:-1, :-1]

        view_zenith = np.deg2rad(
            DWGlintProcessor.get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Zenith'))[:-1, :-1]
        view_azimuth = np.deg2rad(
            DWGlintProcessor.get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Azimuth'))[:-1, :-1]

        phi = sun_azimuth - view_azimuth
        Tetag = np.cos(view_zenith) * np.cos(sun_zenith) - np.sin(view_zenith) * np.sin(sun_zenith) * np.cos(phi)

        # convert results to degrees
        glint_array = np.degrees(np.arccos(Tetag))
        return glint_array

    @staticmethod
    def create_annotated_heatmap(hm, img=None, cmap='magma', vmin=0.7, vmax=0.9):
        '''Create an annotated heatmap. Parameter img is an optional background img to be blended'''
        fig, ax = plt.subplots(figsize=(15, 15))

        ax.imshow(hm, vmin=vmin, vmax=vmax, cmap=cmap)

        if img is not None:
            ax.imshow(img, alpha=0.6, extent=(-0.5, 21.5, 21.5, -0.5))

        # Loop over data dimensions and create text annotations.
        for i in range(0, hm.shape[0]):
            for j in range(0, hm.shape[1]):
                text = ax.text(j, i, round(hm[i, j], 2),
                               ha="center", va="center", color="cornflowerblue")

        return fig, ax

    @staticmethod
    def nn_interpolate(arr, new_size):
        """
        Vectorized Nearest Neighbor Interpolation
        From post: https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203
        """

        old_size = arr.shape
        row_ratio, col_ratio = np.array(new_size) / np.array(old_size)

        # row wise interpolation
        row_idx = (np.ceil(range(1, 1 + new_size[0]) / row_ratio) - 1).astype(int)
        row_idx[-1] = -1

        # column wise interpolation
        col_idx = (np.ceil(range(1, 1 + new_size[1]) / col_ratio) - 1).astype(int)
        col_idx[-1] = -1

        final_matrix = arr[:, col_idx][row_idx, :]

        return final_matrix

    @staticmethod
    def create_glint_heatmap(rgb, glint_arr, limit_angle):
        no_glint_mask = (glint_arr >= limit_angle) | np.isnan(glint_arr)

        glint_prob = np.where(no_glint_mask, 0, 1 - glint_arr / 100)

        return DWGlintProcessor.create_annotated_heatmap(glint_prob, rgb)

    def save_heatmap(self, folder, filename='glint_heatmap.pdf', dpi=50, brightness=5.):
        rgb = np.stack([self.image.raster_bands['Red'],
                        self.image.raster_bands['Green'],
                        self.image.raster_bands['Blue']],
                       axis=2) * brightness

        fig, ax = DWGlintProcessor.create_glint_heatmap(rgb, self.glint_array, self.limit_angle)
        ax.axes.set_axis_off()

        fig.suptitle('GLINT PREDICTION GRID', fontsize=24)
        ax.set_title(f'Image: {self.image.current_image_name}', fontsize=14)

        fn = Path(folder)/filename
        fig.savefig(fn, dpi=dpi)
        fig.clf()

        return fn.as_posix()

    def create_multiplication_coefs(self, min_glint_multiplier=0.5):
        cte = 1.5 + min_glint_multiplier
        return np.where(self.glint_array > self.limit_angle, 0, self.glint_array * -0.05 + cte).astype('float32')

    def show_multiplication_coefs(self):
        return DWGlintProcessor.create_annotated_heatmap(self.create_multiplication_coefs(), vmin=1, vmax=3)

    def glint_adjusted_threshold(self, band, value, thresh_type, mask=None, min_glint_multiplier=0.5):
        """Create an array with the image resolution, with the threshold adjusted for the GLINT
        thresh_type can be SUP or INF """

        # check if it is possible to ajust the threshold. If it is not, return the plain value
        # check the following conditions:
        # 1- if the band should be adjusted
        # 2- if the glint_arry do exists (it can have an error)
        # 3- if there is any possible glint in the scene
        if (band not in self.adjustable_bands) or \
                (self.glint_array is None) or \
                (np.nanmin(self.glint_array) > self.limit_angle):

            return value

        # create a grid with the thresholds, depending on the thresh_type
        # positive value and type sup
        delta_grid = self.create_multiplication_coefs(min_glint_multiplier=min_glint_multiplier) * value

        if value < 0 and thresh_type == 'SUP':
            thresh_grid = value - delta_grid

        if value > 0 and thresh_type == 'SUP':
            thresh_grid = value + delta_grid

        if value < 0 and thresh_type == 'INF':
            thresh_grid = value + delta_grid

        if value > 0 and thresh_type == 'INF':
            thresh_grid = value - delta_grid

        thresh_array = DWGlintProcessor.nn_interpolate(thresh_grid, (self.image.y_size, self.image.x_size))
        return thresh_array[~mask] if mask is not None else thresh_array

    def __repr__(self):
        s = f'Glint Processor for image: {self.image.current_image_folder}'
        return s
