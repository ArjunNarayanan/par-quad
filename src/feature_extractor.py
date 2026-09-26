from src.transformer_feature_extractor import TransformerFeatureExtractor
from src.convolution_feature_extractor import ConvolutionFeatureExtractor


def feature_extractor_initializer(config):
    feature_extractor_config = config["feature_extractor"]
    features_extractor_kwargs = feature_extractor_config.copy()

    feature_extractor_type = features_extractor_kwargs.pop("type")
    if feature_extractor_type == "Transformer":
        sequence_length = config["environment"]["template_size"]
        if "sequence_length" in features_extractor_kwargs:
            assert features_extractor_kwargs["sequence_length"] == sequence_length
        else:
            features_extractor_kwargs["sequence_length"] = sequence_length
        return TransformerFeatureExtractor, features_extractor_kwargs

    elif feature_extractor_type == "Convolution":
        return ConvolutionFeatureExtractor, features_extractor_kwargs
    
    else:
        raise TypeError("Unexpected feature extractor type : ", feature_extractor_type)
