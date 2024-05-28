"""Utility functions"""

from typing import List, Set, Any


class WDCloudUtils:
    """Utility class for working with WD Cloud"""

    @staticmethod
    def flatten_list(nested_list: List) -> List[Any]:
        """Flatten a nested list into a single list"""
        flattened = []
        for item in nested_list:
            if isinstance(item, list):
                flattened.extend(WDCloudUtils.flatten_list(item))
            else:
                flattened.append(item)
        return flattened

    @staticmethod
    def get_unique_values(nested_list: List) -> Set[Any]:
        """Get unique values from a nested list"""
        flattened = WDCloudUtils.flatten_list(nested_list)
        return set(flattened)
