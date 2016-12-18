# Recognising faces using Vision API, TensorFlow & Google Cloud ML

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

The code in `train_local.py` and `train.py` assumes that you have uploaded your training data tarballed, gzipped and uploaded to a GCS bucket.

Check `get_data.sh` to get an idea of how things work.  Better documentation (hopefully) to come!
