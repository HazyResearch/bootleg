"""Emmental dataset and dataloader."""
import logging
from typing import Any, Dict, Optional, Tuple, Union

from emmental import EmmentalDataset
from torch import Tensor

logger = logging.getLogger(__name__)


class RangedEmmentalDataset(EmmentalDataset):
    """
    RangedEmmentalDataset dataset.

    An advanced dataset class to handle that the input data contains multiple fields
    and the output data contains multiple label sets.

    Args:
      name: The name of the dataset.
      X_dict: The feature dict where key is the feature name and value is the
        feature.
      Y_dict: The label dict where key is the label name and value is
        the label, defaults to None.
      uid: The unique id key in the X_dict, defaults to None.
      data_range: The range of data to select.
    """

    def __init__(
        self,
        name: str,
        X_dict: Dict[str, Any],
        Y_dict: Optional[Dict[str, Tensor]] = None,
        uid: Optional[str] = None,
        data_range: Optional[list] = None,
    ) -> None:
        """Initialize RangedEmmentalDataset."""
        super().__init__(name, X_dict, Y_dict, uid)
        if data_range is not None:
            self.data_range = data_range
        else:
            self.data_range = list(next(iter(self.X_dict.values())))

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Tensor]], Dict[str, Any]]:
        """Get item by index after taking range into account.

        Args:
          index: The index of the item.
        Returns:
          Tuple of x_dict and y_dict
        """
        return super().__getitem__(self.data_range[index])

    def __len__(self) -> int:
        """Total number of items in the dataset."""
        return len(self.data_range)
