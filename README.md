# Recognising faces using Vision API, TensorFlow & Google Cloud Machine Learning

This is a work in progress.  Things to come:

1. Better / any documentation :)
2. A prediction demo that uses the trained model

`tf/face_extract/pubfig_get.py` - Use this to download one of the PubFig datasets.  I used the evaluation dataset and split this into training and validation.  It will also crop faces using the supplied PubFig metadata, but note that the face vertices in PubFig can be inaccurate.


`tf/face_extract/crop_faces.py` - Crop faces using the Vision API.  You need a Google Cloud Platform account with Vision API enabled, and a service account.

`tf/pubfig/train.py` - Trains the TensowFlow model.  Use this to train the model locally and have output printed to stdout.  Train locally using:

```
gcloud beta ml local train --package-path=pubfig --module-name=pubfig.train_log
```

`tf/pubfig/train_local.py` - Trains the TensorFlow model, but uses Python logging instead of print statements.  Use this to train on Cloud ML using something similar to:

```
gcloud beta ml jobs submit training pubfig7 --package-path=pubfig --module-name=pubfig.train_log --region=us-central1 --staging-bucket=gs://wwoo-train
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

The code in `train_local.py` and `train.py` assumes that you have uploaded your training data tarballed, gzipped and uploaded to a GCS bucket.

Check `get_data.sh` to get an idea of how things work.  Better documentation (hopefully) to come!
