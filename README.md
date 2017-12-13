# Classifying faces using Vision API, TensorFlow & Google Cloud ML Engine 

## Demo

Live demo (hopefully) [here](http://pubfig-ml.appspot.com/).

Note: the first prediction might come back a little slow.

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
For more details on the python scripts called by `get_data.sh`, see the "Getting and preparing the training data" section below.

Move the prepared data to `/tmp`, where the training code expects to find them by default:
```
$> mv data /tmp
```
When training the model, make sure to specify the correct number of training classes (`--num_classes`) and number of samples in your validation set (`--valid_batch_size`).  This will differ depending on the number of files you've downloaded and how the data has been divided.

Check the training source for other flags you can specify.

### Training a model locally

```
$> cd tf
$> gcloud ml-engine local train \
      --package-path=pubfig \
      --module-name=pubfig.task \
      -- \
      --num_classes=<number_of_classes> \
      --valid_batch_size=<number_of_validation_samples>
$> #
$> # lots of output follow
```
Your trained model will be exported to `/tmp/model/` by default.

## Training Quickstart on Cloud ML Engine

Install the Cloud ML SDK and prepare your environment per the [setup guide](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

### Training a model using Cloud ML to serve using ML Engine Online Prediction
Example:
```
$> gcloud ml-engine jobs submit training example_job123 \
      --package-path=pubfig \
      --module-name=pubfig.task \
      --region=us-central1 \
      --staging-bucket=gs://wwoo-train \
      --runtime-version=1.2
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
14:20:45.629 get_image_label_list: read 336 items
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

## Training Results
The model can be trained to 80% validation accuracy with 48 classes (face categories), using 4402 training and 336 validation samples.  With the default hyperparameters, overfitting started to occur past ~1.2K steps using a learning rate of 0.01.

Orange = training set, Blue = validation set.

Cross entropy loss:

![Loss](https://storage.googleapis.com/wwoo-htdocs/images/tf_face_xent.png "Loss")

Classification accuracy:

![Accuracy](https://storage.googleapis.com/wwoo-htdocs/images/tf_face_acc.png "Accuracy")

## Getting and preparing the training data

This section describes the Python scripts called by `get_data.sh`.  Provided everything completed successfully, you should not need to call these scripts directly to retrieve and process the data.

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

## Web Interface

### Using Cloud ML Engine Online Prediction

`tf_face/web/` contains source which can be deployed to Google App Engine.

You will need to modify `tf_face/web/main.py` to match your project ID and model name.  Also replace `resources/vapi-acct.json.replaceme` with your service account key.

Deploy using:
```
$> cd ${SRC_ROOT}/tf/web
$> gcloud app deploy
$>
$> # lots of output follows
```
