# Recognising faces using Vision API, TensorFlow & Google Cloud Machine Learning

## Demo

Live demo (hopefully) [here](http://104.196.149.8:8080).

## Training Quickstart (local)

Firstly, follow the [setup guide](https://cloud.google.com/ml/docs/how-tos/getting-set-up) to install the Google Cloud Machine Learning SDK.  This will also ask you to install TensorFlow.

Set environment variables:
```
$> SRC_ROOT=wherever/you/cloned/the/files
```
Install prerequisites:
```
$> cd $SRC_ROOT
$> pip install -r requirements.txt
```
Download and prepare the data:
```
$> # NOTE: you need a GCP service account saved as $SRC_ROOT/tf/face_extract/vapi-acct.json
$> # to call Google Cloud Vision API
$> #
$> ./get_data.sh
$> #
$> # lots of output follows
```
Move the prepared data to `/tmp`, where the training code expects to find them by default:
```
$> mv data /tmp
```
Train the model locally:
```
$> cd tf
$> gcloud beta ml local train --package-path=pubfig_export --module-name=pubfig_export.export
$> #
$> # lots of output follow
```
Your trained model will be exported to `/tmp/model/00000001` by default.
```
$> ls /tmp/model/00000001
checkpoint  export.data-00000-of-00001  export.index  export.meta
$>
```

## Prediction Quickstart (local)

Firstly, you will need to install TensorFlow Serving by following the guide [here](https://tensorflow.github.io/serving/setup). This will also ask you to install the Bazel build system.

Set environment variables:
```
$> SRC_ROOT=wherever/you/cloned/the/files
$> TF_SERVING_ROOT=wherever/you/cloned/tensorflow/serving
```
Install the prerequisites:
```
$> cd $TF_SERVING_ROOT/web
$> pip install -t lib -r requirements.txt
```
Bazel build and run the prediction server:
```
$> cd $TF_SERVING_ROOT
$> ln -s $SRC_ROOT tf_models/tf_face
$> bazel build tf_models/tf_face/tf/web/predict_serving
$> # ... output
$> /bazel-bin/tf_models/tf_face/tf/web/predict_serving &
```
Serve the model:
```
$> ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=pubfig --model_base_path=/tmp/model
```
Navigate to http://localhost:8080.

## Getting and preparing the training data

The model is trained using a subset of data from [PubFig](http://www.cs.columbia.edu/CAVE/databases/pubfig/). PubFig provides a development set and evaluation set of images, with no people or sample overlaps between the two.  For our face recognition use case, we will just use the evaluation dataset and split these further into training and validation.

PubFig provides only the links to images on the public web, not the images themselves.  Therefore, it is necessary to download them separately using `pubfig_get.py`.  Some images may not be downloaded successfully due to technical issues such as broken links, removed content, unreachable servers and so-on.

### Downloading the data

`tf/face_extract/pubfig_get.py` - Use this to download one of the PubFig datasets.  I used data from the [evaluation set](http://www.cs.columbia.edu/CAVE/databases/pubfig/download/#eval). Save this as `eval_urls.txt` in the same directory as `pubfig_get.py`.  `pubfig_get.py` will also crop faces using face vertices supplied in the PubFig metadata, but I encountered incorrect vertices in some cases. Set `IMAGE_CROP = True` if you want the script to crop out the faces using the supplied PubFig metadata.  

Example invocation to read from `eval_urls.txt` and save to the `./data` directory:

```
$> python tf/face_extract/pubfig_get.py tf/face_extract/eval_urls.txt ./data
```

`pubfig_get.py` generates a `manifest.txt` file, which is a list of local file paths to the downloaded files.  You will likely see some duplicates, due to conflicting filenames that `pubfig_get.py` would have overwritten [TODO - fix].  Remove the duplicates using:

```
$> cat ./data/manifest.txt | sort | uniq > ./data/manifest_uniq.txt
```

### Cropping the faces

`tf/face_extract/crop_faces.py` - Crop faces using Google Vision API, which is far more accurate than using the PubFig supplied metadata.  You need to sign up for a Google Cloud Platform account to use the Vision API.  

Follow the Vision API [Quickstart](https://cloud.google.com/vision/docs/quickstart) to enable the Vision API on your Google Cloud Platform project.  You will also need to [generate a service account](https://cloud.google.com/storage/docs/authentication#generating-a-private-key) that `crop_faces.py` can use to call the Vision API with.  Save the service account JSON as `vapi-acct.json` in the same directory as `crop_faces.py`.

Cropped files are saved in a `crop` directory in the same parent directory as the original file.

Example invocation to crop all files from the paths in `manifest_uniq.txt`, with the current working directory prepended to each file path (as paths are relative in `./data/manifest_uniq.txt`):

```
$> python tf/face_extract/crop_faces.py ./data/manifest_uniq.txt $PWD
```

*Note:* Using Vision API will cost you money, though you can always sign up for a free Google Cloud Platform account with $300 in credits.  It is your responsibility to manage your own usage.

### Dividing the data into training and validation sets

`tf/face_extract/split_data.py` - Splits the cropped data into training and validation sets.  You can adjust a number of factors in splitting the data - for example, the ratio of training to validation data (`SPLIT_FACTOR`), the minimum & maximum samples for a class to be included (`MIN_SAMPLES` and `MAX_SAMPLES`) and set a minimum/maximum skew (roll, pan and tilt) for a sample to be included [TODO: expose in more friendly way].

Example invocation to read from `./data/vision-manifest.txt`, and write the training and validation dataset (as a set of paths) to `train.txt` and `valid.txt`.

```
$> python tf/face_extract/split_data.py ./data/vision-manifest.txt ./data/train.txt ./data/valid.txt
```

## Training the model

You can train and export a model using Google Cloud Machine Learning, or using TensorFlow.

Whichever you choose, you need to ensure that your input and output paths are set correctly.  See the source for more details [TODO: more details on specific flags to use].

A pre-trained model that you can use exists at `tf/sample_run/models/00000001`.

### Cloud Machine Learning

To use Cloud Machine Learning, you need to have a Google Cloud Platform project with the service activated and billing enabled.

*Note:* Using Cloud Machine Learning will cost you money, though you can always sign up for a free Google Cloud Platform account with $300 in credits.  It is your responsibility to manage your own usage.

Follow the [Cloud Machine Learning setup guide](https://cloud.google.com/ml/docs/how-tos/getting-set-up) to install all the local pre-requisites.

`tf/pubfig/train.py` - Trains the TensowFlow model.  Use this to train the model locally using the gcloud SDK and have output printed to stdout.

```
$> gcloud beta ml local train --package-path=pubfig --module-name=pubfig.train_log
```

`tf/pubfig/train_local.py` - Similar to `train.py`, but uses Python logging instead of print statements.  Use this to train on Cloud ML using something similar to:

```
$> gcloud beta ml jobs submit training pubfig7 --package-path=pubfig \
$>    --module-name=pubfig.train_log --region=us-central1 --staging-bucket=gs://wwoo-train
```

The job output will be similar to the below. In this case, training terminates once 75% validation accuracy is reached.

```
14:13:29.600 Validating job requirements...
14:13:31.063 Job creation request has been successfully validated.
14:13:31.183 Job pubfig7 is queued.
14:13:36.350 Waiting for job to be provisioned.
14:20:35.038 Running command: python -m pubfig.train_log
14:20:36.915 Recursively copying from gs://wwoo-train/pubfig/out.tar.gz to /tmp/
14:20:45.528 get_image_label_list: read 4416 items
14:20:45.629 get_image_label_list: read 384 items
...
18:42:03.778 Step [6700] (valid): accuracy: 0.703125, loss: 3.42011
18:45:55.734 Step [6800] (train): accuracy: 0.729167, loss: 3.08664
18:46:04.288 Step [6800] (valid): accuracy: 0.708333, loss: 3.33873
18:49:49.276 Step [6900] (train): accuracy: 1.0, loss: 0.0014053
18:49:57.623 Step [6900] (valid): accuracy: 0.752604, loss: 2.98494
18:49:57.624 Step [6900] (complete)
18:49:57.866 Recursively copying from /tmp/logs to gs://wwoo-train/pubfig/export/
18:49:59.649 Recursively copying from /tmp/model to gs://wwoo-train/pubfig/export/
18:50:01.943 Module completed; cleaning up.
18:50:01.944 Clean up finished.
18:50:01.945 Task completed successfully.
18:50:25.382 Tearing down TensorFlow.
18:51:09.881 Finished tearing down TensorFlow.
18:51:16.318 Job completed successfully.
```

### TensorFlow

To serve your model using TensorFlow Serving, use `export.py` to train and export your model.

```
$> gcloud beta ml local train --package-path=pubfig_export --module-name=pubfig_export.export
```

## Web Interface

### Using Cloud Machine Learning online prediction

[TODO] Add source and instructions

### Using TensorFlow Serving

TensorFlow Serving comes with a standard model server.  You can run it using:

```
$> $TF_SERVING_ROOT/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
$>   --port=9000 --model_name=pubfig --model_base_path=sample_run/models/
```

The web interface uses the TensorFlow Serving protos, so the easiest way to run it is again symlink'ing the source to wherever you build TensorFlow Serving.  For example, if you symlink'ed `tf_face` to `$TF_SERVING_ROOT/tf_models/tf_face`:

```
$> # Build it
$> bazel build $TF_SERVING_ROOT/tf_models/tf_face/tf/web/predict_serving

$> Run it
$> $TF_SERVING_ROOT/tf_models/tf_face/tf/web/predict_serving
```
