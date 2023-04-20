
import numpy as np
from keras.applications import vgg16,vgg19, resnet
import tensorflow as tf
from keras.models import Model


def vgg16FeatureExtract(image_batch):
    # load VGG16 model
    vgg_model = vgg16.VGG16(weights='imagenet')

    # We removed the last layer because we want to get features instead of predictions
    feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    #feature_extractor.summary()

    # extracting image features
    vgg16_features = feature_extractor.predict(image_batch)

    return vgg16_features

def vgg19FeatureExtract(image_batch):
    # load VGG19 model
    vgg_model = vgg19.VGG19(weights='imagenet')

    # We removed the last layer because we want to get features instead of predictions
    feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    #feature_extractor.summary()    

    # extracting image features
    vgg19_features = feature_extractor.predict(image_batch)

    return vgg19_features

def resnetFeatureExtract(image_batch):
    # load resnet model
    resnet_model = resnet.ResNet50(weights='imagenet')

    # We removed the last layer because we want to get features instead of predictions
    feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.get_layer("avg_pool").output)
    #feature_extractor.summary()

    # extracting image features
    resnet_features = feature_extractor.predict(image_batch)

    return resnet_features

def inceptionV3FeatureExtract(image_batch):
    # load resnet model
    inception_model = tf.keras.applications.InceptionV3(weights='imagenet')

    # We removed the last layer because we want to get features instead of predictions
    feature_extractor = Model(inputs=inception_model.input, outputs=inception_model.get_layer("avg_pool").output)
    #feature_extractor.summary()

    # extracting image features
    inception_features = feature_extractor.predict(image_batch)

    return inception_features