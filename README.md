# toxic_comment_detector
## Build Environment
The environment "toxic" can be built by the following command.
```cmd
conda env create -f environment.yaml
```  

Activate environment by
```cmd
conda activate toxic
```


## Start web server
The web server can be activated by the following command.
```cmd
# activate env
conda activate toxic
# change to webside directory
cd toxic_comment
# run server
python manage.py runserver
```

or directly access to folder toxic_comment and run `activate.sh`

## Deep learning model
Retrain
```cmd
# change to folder "model_training"
cd model_training

# start training
# --kaggle_submit : generate kaggle submission result
# --save_model_tokenizer : save the training model
python train.py --kaggle_submit --save_model_tokenizer
```

The original training and evalutation data could be found at following websites.
* https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview
* https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset