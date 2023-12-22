from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch as th
from pandas import DataFrame

from ..containers import BaseContainer, TabularContainer
from ..extractors import FeatureExtractor, TabularFeatureExtractor
from ..iterators import BaseIterator
from ..preprocessors.base_preprocessor import FeaturePreprocessor
from ..utils.utils import DatasetType


@dataclass
class FeaturePipeline:
    extractor: FeatureExtractor
    """Tuple defining the feature extractor and its initialization arguments."""
    preprocessors: Optional[List[FeaturePreprocessor]] = None
    """List of tuples defining feature preprocessors
    and their initialization arguments."""


IT = TypeVar("IT", bound=BaseIterator)


class DataProcessor(Generic[IT]):
    """
    Parameters
    ----------
    iterator_cls: Type[BaseDatasetIterator]
        Dataset iterator class type.

    feature_extractors: Mapping[str, Tuple[FeatureExtractorType, Any]]
        Dict that links a feature name to tuples of the form (Extractor, args)
        where Extractor implements FeatureExtractor and args are the arguments
        to be passed at initialization.
        N.B.: Feature names must match the iterator's parameters.

        If Extractor is a StaticExtractor, the features are extracted
        independently of the split.

        If Extractor is a TrainedExtractor, the extractor is first fitted
        on the train split and then applied to the other splits.

    feature_preprocessors: Optional[Mapping[str, List[FeaturePreprocessor]]]
        Dict that links a feature name to a list of tuples of the form (Processor, args)
        where Processor implements FeatureProcessor and args are the arguments
        to be passed at initialization.
        This allows to apply a series of processors to different features, e.g.
        to normalize the features.
        N.B.: Feature names must match the iterator's parameters.
        If Processor is a StaticExtractor, the features are processed
        independently of the split.

        If Extractor is a TrainedExtractor, the processor is first fitted
        on the train split and then applied to the other splits
        (typically for normalization).

    tensors_dtype: Optional[th.dtype]
        Data type of the tensors returned by the iterator, by default None
    """

    def __init__(
        self,
        iterator_cls: Type[IT],
        feature_extractors: Dict[str, FeatureExtractor],
        feature_preprocessors: Optional[
            Mapping[
                str,
                Sequence[FeaturePreprocessor],
            ]
        ] = None,
        tensors_dtype: Optional[th.dtype] = None,
    ) -> None:
        self.iterator_cls = iterator_cls
        self.feature_extractors = feature_extractors
        self.feature_processors = feature_preprocessors or {}

    def _apply_processing_function(
        self,
        function: Callable[..., BaseContainer],
        data: Union[DataFrame, BaseContainer],
        split: DatasetType,
        is_trained: bool,
    ) -> BaseContainer:
        if is_trained:
            features = function(data, split=split)
        else:
            features = function(data)

        return features

    def extract_features(
        self, data: DataFrame, split: DatasetType
    ) -> Dict[str, BaseContainer]:
        """Extract features for the different splits of the data.

        Returns
        -------
        DataHandler
            self

        Raises
        ------
        ValueError
            Expects data to be split before extracting features.
        """
        features: Dict[str, BaseContainer] = {}
        for name, extractor in self.feature_extractors.items():
            features[name] = self._apply_processing_function(
                extractor.extract_features,
                data,
                split=split,
                is_trained=extractor.trained,
            )
            for preprocessor in self.feature_processors.get(name, []):
                features[name] = self._apply_processing_function(
                    preprocessor.preprocess,
                    features[name],
                    split=split,
                    is_trained=preprocessor.trained,
                )

        return features

    def make_iterator(self, data: DataFrame, keys: Sequence, split: DatasetType) -> IT:
        return self.iterator_cls(keys, **self.extract_features(data, split=split))

    def inverse_transform(
        self, container: TabularContainer, pipeline_name: str
    ) -> DataFrame:
        """Inverse transform the data to the original format.

        Parameters
        ----------
        container: TabularContainer
            Data to be inverse transformed.
        pipeline_name: str
            Name of the feature pipeline to be inverse transformed.
        Returns
        -------
        DataFrame
            Inverse transformed data.
        """

        extractor = self.feature_extractors[pipeline_name]
        if not isinstance(extractor, TabularFeatureExtractor):
            raise ValueError(
                "Only TabularFeatureExtractor supports"
                "transforming back to original dataframe."
            )
        preprocessors = self.feature_processors.get(pipeline_name, [])

        for preprocessor in preprocessors[::-1]:
            if not hasattr(preprocessor, "inverse_transform"):
                raise ValueError(
                    f"Feature preprocessor {pipeline_name} does "
                    "not have an inverse transform method."
                )
            container = preprocessor.inverse_transform(container)  # type: ignore
        df = cast(TabularContainer, container).data
        return df


def create_data_processor(
    iterator_cls: Type[IT], *args: str
) -> Callable[..., DataProcessor[IT]]:
    """
    Creates a function dynamically to instatiate DataProcessor
    based on provided iterator class and additional arguments.

    Parameters
    ----------
    iterator_cls : Type[BaseDatasetIterator]
        Dataset iterator class type.
    args : str
        Additional feature names to be included.

    Returns
    -------
    create_data_processor: Callable[..., DataProcessor]
        A dynamically generated function
        with arguments derived from the provided iterator class,
        in addition to other specified arguments.

    Notes
    -----
    The returned function has the following signature:
        >>> def get_processor(
        >>>    **kwargs: FeaturePipeline,
        >>> ) -> DataProcessor:

        where kwargs are the feature names and their corresponding feature
    """
    params = iterator_cls.get_parameter_names()
    if args is not None:
        params += args

    def get_processor(
        tensors_dtype: Optional[th.dtype] = None,
        **kwargs: FeaturePipeline,
    ) -> DataProcessor[IT]:
        feature_extractors: Dict[str, FeatureExtractor] = {}
        feature_preprocessors: Dict[str, Sequence[FeaturePreprocessor]] = {}
        for param in params:
            if param in kwargs:
                feature_extractors[param] = kwargs[param].extractor
                preprocessors = kwargs[param].preprocessors
                if preprocessors is not None:
                    feature_preprocessors[param] = preprocessors
            else:
                raise ValueError(
                    f"Feature pipeline for {param} not specified in kwargs. "
                    f"All iterator features should be provided with an extractor."
                )

        return DataProcessor[IT](
            iterator_cls=iterator_cls,
            tensors_dtype=tensors_dtype,
            feature_extractors=feature_extractors,
            feature_preprocessors=feature_preprocessors,
        )

    return get_processor
