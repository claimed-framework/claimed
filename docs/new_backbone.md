# Benchmarking a new backbone

The benchmarking tool is built on TerraTorch, so natively has access to all backbones available in TerraTorch by specifying them by name in the `backbone` key.

However, there may be instances where we want to evaluate a backbone which is not in the TerraTorch library. There are a few ways to achieve this.

## 1. Using a different set of weights

The `PrithviModelFactory` leverages timm underneath to fetch models when a string is passed as the backbone. As such, the `backbone_args` are also passed to the `timm.create_model` method.

To use a new set of weights, we can therefore use the same strategy we would in timm, namely by passing the `pretrained_cfg_overlay` key to the `backbone_args`:

```yaml
backbone:
    backbone_args:
        pretrained_cfg_overlay:
            file: <path to weights>
```

Other factories may have other methods of passing pretrained weights paths.

## 2. Using a different model architecture

If you are using the `PrithviModelFactory`, there are two ways you can add new backbones.

### 1. Pass the backbone instance directly to the model factory

The `PrithviModelFactory`, despite its name, is actually quite versatile and makes few assumptions about the backbones it can support.

As such, instead of passing a string with the name of the backbone, you can pass it an instance of the actual backbone.

There are, however, two conditions that must be fulfiled:

### Feature info

The model must have a `feature_info` attribute which is an instance of `timm.models.FeatureInfo`.
This collects information about the output shape of the features of the model. For each eligible output the model may return, a dictionary with three features must be collected in a list: `{"num_chs": in_chans, "reduction": 4 * scale, "module": f"stages.{i}"}`.
These hold the embedding dimension, the spatial reduction with respect to the initial image, and the name of the module where it comes from.

In reality, we only rely on the "num_chs" key, so the others may be left as dummy values.
The other argument when constructing FeatureInfo is the indices of these features which are actually output by the model.

Usually, each of these dicts can be constructed when each block of the model is being initialized and accumulated into a list.
This can then be used to instantiate `FeatureInfo` at the end of the constructor.

### Feature transformation

The output of your model should be suitable for whichever decoder you will plug into it. If it is not, you may define a function `prepare_features_for_image_model` that takes the model output and reshapes it in whichever way will make it compatible with the decoder.

This function may be a method of your backbone or it may be passed to the `PrithviModelFactory` with that argument name.

### Using the model

Once these steps are complete, you may instantiate your model in the config file and pass it to the `PrithviModelFactory`


## 2. Register the model with timm

Alternatively, you may register the backbone with timm. This involves some more work, but makes your model ready to be added to TerraTorch if so desired.

You can see examples of this in the `vit_encoder_decoder.py` or `swin_encoder_decoder.py` and `prithvi_vit.py` or `swin_vit.py` classes in TerraTorch or any other model implemented in `timm`. The file `benchmark_new_backbone_timm.yaml` demonstrates how to use it in a config file.

Once this is done, you just need to point the benchmarking tool to the module that registers your new backbones with the key `backbone_registration`. You can now use your backbone as any of the backbones in TerraTorch.

### Example

You can see an example of this for the Swin3D model in the `swin3d` directory. Specifically files `prithvi_swin3d.py` for registering with timm and `swin3d.py` for adding `feature_info`.

## 3. Using a different model factory

A third alternative is to use a different model factory altogether, specifying how to build the end to end model yourself.
This needs to be registered as a function in TerraTorch.

This option is less flexible than the others, but may be the simplest depending on your scenario.
