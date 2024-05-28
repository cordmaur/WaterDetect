"""WaterdetectCloud Engine module"""
from pathlib import Path
from typing import Set, Dict, Optional, List
import numpy as np

import matplotlib.pyplot as plt

import waterdetect as wd
from waterdetect.Common import DWutils
from .tile import ImgTile
from .utils import WDCloudUtils


class WDCloudEngine:
    """
    The WDCloudEngine class is the main engine of the waterdetect-cloud library.
    It is responsible for loading the necessary bands, running the clustering algorithm,
    and generating the graphs.
    """
    # ########## Initialization Methods ##########
    def __init__(
        self,
        img: ImgTile,
        config_file: str,
        resolution: Optional[int] = None,
    ):
        # Load the config file
        self.config = wd.DWConfig(config_file)

        # Initialize the variables
        self.clusters = self.water = self.graphs = None
        self.img = img

        if resolution is not None:
            self.img.resolution = resolution

        self.bands = self.load_bands()

        # get the invalid mask for this scene
        self.mask = self.load_mask()

    def load_mask(self) -> np.ndarray:
        """
        Load the invalid mask for this scene, considering the MASKS Section from the config file.
        """

        # first get the masks from the config file
        masks = self.config.get_masks_list(self.img.metadata["product_type"])

        print(f"Combining the following masks {str(masks)}")
        return self.img.get_mask(masks).data

    def check_necessary_bands(self) -> Set[str]:
        """
        Check all the bands mentioned in the config file and return a set of the valid bands.
        """
        # first, let's grab all bands/indices mentioned in the config
        # these are common names
        s = set([self.config.detect_water_cluster][3:])
        s = s.union(WDCloudUtils.get_unique_values(self.config.clustering_bands))
        s = s.union(WDCloudUtils.get_unique_values(self.config.graphs_bands))

        # Then, check if they are valid and split the indices
        bands = set()
        for common_name in s:
            # check if the common name is a valid band name
            if common_name in self.img.metadata["bands_names"]:
                bands.add(common_name)

            # Check if the common name is a valid index
            elif common_name in DWutils.indices:
                # add the index to the list of bands
                for band in DWutils.indices[common_name]:
                    bands.add(band)

            else:
                raise ValueError(f"Invalid common name: {common_name}")

        return bands

    def load_bands(self) -> Dict[str, np.ndarray]:
        """Load the necessary bands into a dictionary as numpy arrays"""
        bands_dict = {}

        common_names = self.check_necessary_bands()
        bands = self.img.convert_common_names(common_names)

        print(f"Loading bands: {common_names}\nOutput shape {self.img.shape}")
        bands_dict = {
            common_name: self.img[band].data
            for common_name, band in zip(common_names, bands)
        }

        return bands_dict

    # ########## Private Methods ##########
    def _detect_water(self, bands_combination: List):
        """
        Detect water in the scene for one simple bands_combination and keeps result in memory.
        """

        # run the clustering algorithm
        wdetect = wd.DWImageClustering(
            bands=self.bands,
            bands_keys=bands_combination,
            invalid_mask=self.mask,
            config=self.config,
        )

        # Run the main algo 
        self.clusters = wdetect.run_detect_water()
       
        # Adjust the mask accordingly
        self.clusters[self.mask] = 0
        self.mask |= (self.clusters == 0)

        # Create the output water mask
        self.water = (self.clusters == 1).astype('uint8')
        self.water[self.mask] = 255

    # ########## Static Methods ##########
    @staticmethod
    def save_graphs(graphs: dict, output_folder: str):
        
        output_folder = Path(output_folder)
        
        for key, fig in graphs.items():
            fname = f'Graph_{key}.png'
            fig.savefig(output_folder/fname)


    # ########## Plotting Methods ##########
    def plot_graph(self, samples, idxs, bands: List) -> plt.Figure:

        x_band = bands[0]
        y_band = bands[1]

        x_values = self.bands[x_band][~self.mask][idxs]
        y_values = self.bands[y_band][~self.mask][idxs]

        # plot the samples
        fig, ax = plt.subplots()

        for cluster in np.unique(samples.astype('uint8')):
            label = 'Water' if cluster == 1 else f'Cluster {cluster}'
            ax.scatter(
                x_values[samples == cluster], 
                y_values[samples == cluster],
                s=1,            
                label=label
            )
            
        # Retrieve the handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()

        # Create a new legend with larger markers
        ax.legend(handles, labels, loc='upper right', markerscale=5)

        ax.set_ylabel(y_band)
        ax.set_xlabel(x_band)

        ax.set_title(bands)

        return fig

    def generate_graphs(self, graph_bands: List):
        self.graphs = {} 

        n_samples = 1000

        # grab just valid pixels  into a single vector
        clusters = self.clusters[~self.mask]

        # get the indices of the pixels that will be considered
        n_samples = n_samples if n_samples < len(clusters) else len(clusters)
        idxs = np.random.randint(0, len(clusters), size=n_samples)

        # grab the samples 
        samples = clusters[idxs]

        # Switch to a non-interactive backend
        current_backend = plt.get_backend()
        plt.switch_backend('agg')

        # adjust the graph_bands in case someone pass just one combination
        if isinstance(graph_bands[0], str):
            graph_bands = [graph_bands]

        for bands in graph_bands:
            print(f'Generating graph: {bands}')
            self.graphs['_'.join(bands)] = self.plot_graph(samples, idxs, bands)
        
        plt.switch_backend(current_backend)

        print("Graphs are available in `.graphs` attribute.")

    # ########## Public Methods ##########
    def detect_water_single(self, bands_combination: List, output_folder: Optional[str] = None):

        # First, ran the main method with the specific bands_combination
        self._detect_water(bands_combination)

        # Then, generate the graphs according to the bands in the config file
        self.generate_graphs(graph_bands = self.config.graph_bands, output_folder=output_folder)






