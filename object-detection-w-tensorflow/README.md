# Object Detection with TensorFlow

In this demo, we'll build an object detector and classifier via Transfer Learning. 



## Preparing the training data



## Converting images to TensorFlow Records






## Training
### Base Model
Choose a pre-trained model from the [TF Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Currently you can find models pre-trained on: 
- [COCO dataset](http://mscoco.org)
- [Kitti dataset](http://www.cvlibs.net/datasets/kitti/)
- [Open Images dataset](https://github.com/openimages/dataset)
- [AVA v2.1 dataset](https://research.google.com/ava/)
- [iNaturalist Species Detection Dataset](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#bounding-boxes)

These models can be used out-of-the-box inference in the categories already in those databases. In our case we'll use them for initializing our model on our dataset. A process called as Transfer Learning.  

Below is a list of some of the models to consider. Each one varies in accuracy and speed. 

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 30 | 21 | Boxes |
| [ssd_mobilenet_v1_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29 | 18 | Boxes |
| [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | 31 | 22 | Boxes |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58 | 28 | Boxes |
| [faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz) | 82 |  | Boxes |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) | 89 | 30 | Boxes |


For this demo, I'll use `ssd_mobilenet_v1_coco`. Download and un-tar the file:
```Console
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -xzvf ssd_mobilenet_v1_coco.tar.gz
```

Inside the un-tar'ed directory, you will find:

* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference
* a config file (`pipeline.config`) which was used to generate the graph.  These
  directly correspond to a config file in the
  [samples/configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)) directory but often with a modified score threshold.  In the case
  of the heavier Faster R-CNN models, we also provide a version of the model
  that uses a highly reduced number of proposals for speed.


Note: 
- The asterisk (☆) at the end of model name indicates that this model supports TPU training.
- The frozen inference graphs from the models above are generated using the
  [v1.8.0](https://github.com/tensorflow/tensorflow/tree/v1.8.0). 
- It's common that you may be working with an older version of TF. In that case you can regenerate the frozen inference graph to your current version by using the
  [exporter](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md) tool,
  pointing it at the model directory as well as the corresponding config file in
  [samples/configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

### Configuration File

The **config** file is dependent of the model and the data used, so it requires some [adjustments](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

At a high level, the **config** file is divided in 5 sections: 
1. `model`: what type of model will be trained (i.e.: meta-architecture, feature extractor)
2. `train_config`: what parameters should be used to train the model (i.e.: SGD params, input preprocessing, feature extractor init values)
3. `eval_config`: what set of metrics will be reported for evaluation. 
4. `train_input_config`: what dataset the model should be trained on.
5. `eval_input_config`: what dataset the model will be evaluated on. 


Some parameters you may want to adjust from the [model config template](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs): 
- Specify the `input_path` to the data TFRecord files for training and evaluation, the `label_map_path`.
- Define and the path to the `fine_tune_checkpoint`. This is the model parameter initialization. The model re-uses the feature extractor parameters from the pre-existing model checkpoint. `from_detection_checkpoint` is a boolean; if false indicates the checkpoints is from an object classification checkpoint.
- Uptade the `num_classes`
- Configure the `ssd_anchor_generator`: 
    - `min_scale`, `max_scale` for the bounding boxes
    - `aspect_ratios`: width/height of the bounding boxes
- In train_config, adjust the `num_steps`
- In eval_config, adjust `num_examples`
- In feature_extractor, adjust `max_detections_per_class` an `max_total_detections`


### Training the model

I've provided a notebook that can be run in Colab with the following steps:

1. Download the trained model and corresponding config file

```
python object_detection/model_main.py 
            --pipeline_config_path=path/to/the/model/config 
            --model_dir=path/to/the/output
```

Side note on [`checkpoints`](https://developers.google.com/machine-learning/glossary/#checkpoint):
- By definition: *a checkpoint is data that captures the state of the variables of a model at a particular time. They enable exporting model weights, as well as performing training across multiple sessions.
- Restoring a model's state from a checkpoint only works if the model and checkpoint are compatible.
- By default, during training an `Estimator` saves checkpoints in the `model_dir` every 600 sec, at the start (first iteration) and at the end (final iteration); and only retains the 5 most recent.
- The default values can be altered using [tf.estimator.RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig). i.e.:
```
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
)
```

...


## Export the Inference Graph

In order to use the model for inference in production, the graph must be freezed. TensorFlow provides an utility to export the frozen model to a TF graph proto: [Exporting Trained Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)



