
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import silhouette_score
from lowLevelFeatures import *
from highLevelFeatures import *
from clusteringAlgos import *

directory = ".\style"
count = 0
low_level_features = []
image_array = []
image_array_ = []

style = pd.read_csv ('style\style.csv')
style = style[style["file"] != '0_0_081.png']
style = style[style["file"] != '4_0_036.png']
results = pd.DataFrame()

#-----------------------------------------------------------------------------------------
# Feature Extraction from images
#-----------------------------------------------------------------------------------------

for prod in set(style["product_label"]):
    
    sub_style = style.loc[style['product_label'] == prod, ]
    count = 0
    low_level_features = []
    image_array = []
    image_array_ = []

    for file in sub_style['file']:
        count = count + 1
        print(prod, "-", count)

        image = cv2.imread("./style/" + file)
        rgb_hist = rgbHist(image, num_bins = 20)
        hsv_hist = hsvHist(image, num_bins = 20)
        #hog_feature = HoG(image, num_bins = 20, rescaled = True, max_val = 10)
        # haralick_feature = haralickFeat(image)
        
        low_level_features.append(np.concatenate((rgb_hist, hsv_hist), axis=None))
        
        preproc_image = np.expand_dims(cv2.resize(image, (224, 224)), axis=0)
        image_array.append(preproc_image)
        preproc_image_ = np.expand_dims(cv2.resize(image, (299, 299)), axis=0)
        image_array_.append(preproc_image_)

    # low level features
    low_level_features = np.vstack(low_level_features)
    print("Completed low level features - prod", prod)
    '''
    filename = './results/low_level_features_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(low_level_features, f)
    print("Saved low_level_features to local", prod)
    '''
    # high level features
    image_array = np.vstack(image_array)

    vgg16_features = vgg16FeatureExtract(image_array)
    print("Completed vgg16 features - prod", prod)
    '''
    filename = './results/vgg16_features_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(vgg16_features, f)
    print("Saved vgg16_features to local", prod)
    '''

    vgg19_features = vgg19FeatureExtract(image_array)
    print("Completed vgg19 features - prod", prod)
    '''
    filename = './results/vgg19_features_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(vgg19_features, f)
    print("Saved vgg19_features to local", prod)
    '''

    resnet_features = resnetFeatureExtract(image_array)
    print("Completed resnet features - prod", prod)
    '''
    filename = './results/resnet_features_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(resnet_features, f)
    print("Saved resnet_features to local", prod)
    '''
            

    image_array_ = np.vstack(image_array_)
    inception_features = inceptionV3FeatureExtract(image_array_)
    print("Completed inception features - prod", prod)
    '''
    filename = './results/inception_features_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(inception_features, f)
    print("Saved inception_features to local", prod)
    '''


    #print(low_level_features.shape, vgg16_features.shape, vgg19_features.shape, resnet_features.shape, inception_features.shape)

    #-----------------------------------------------------------------------------------------
    # distance metrics
    #-----------------------------------------------------------------------------------------
    
    # consine similarity
    cos_similarity_low = cosine_similarity(low_level_features)
    filename = './results/cos_similarity_low_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cos_similarity_low, f)
    print("Saved cos_similarity_low to local")

    cos_similarity_vgg16 = cosine_similarity(vgg16_features)
    filename = './results/cos_similarity_vgg16_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cos_similarity_vgg16, f)
    print("Saved cos_similarity_vgg16 to local")

    cos_similarity_vgg19 = cosine_similarity(vgg19_features)
    filename = './results/cos_similarity_vgg19_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cos_similarity_vgg19, f)
    print("Saved cos_similarity_vgg19 to local")

    cos_similarity_resnet = cosine_similarity(resnet_features)
    filename = './results/cos_similarity_resnet._prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cos_similarity_resnet, f)
    print("Saved cos_similarity_resnet to local")

    cos_similarity_inception = cosine_similarity(inception_features)
    filename = './results/cos_similarity_inception_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cos_similarity_inception, f)
    print("Saved cos_similarity_inception to local")

    # eulidean distance
    euclidean_similarity_low = euclidean_distances(low_level_features)
    filename = './results/euclidean_similarity_low_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(euclidean_similarity_low, f)
    print("Saved euclidean_similarity_low to local")

    euclidean_similarity_vgg16 = euclidean_distances(vgg16_features)
    filename = './results/euclidean_similarity_vgg16_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(euclidean_similarity_vgg16, f)
    print("Saved euclidean_similarity_vgg16 to local")

    euclidean_similarity_vgg19 = euclidean_distances(vgg19_features)
    filename = './results/euclidean_similarity_vgg19_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(euclidean_similarity_vgg19, f)
    print("Saved euclidean_similarity_vgg19 to local")

    euclidean_similarity_resnet = euclidean_distances(resnet_features)
    filename = './results/euclidean_similarity_resnet_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(euclidean_similarity_resnet, f)
    print("Saved euclidean_similarity_resnet to local")

    euclidean_similarity_inception = euclidean_distances(inception_features)
    filename = './results/euclidean_similarity_inception_prod_' + str(prod) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(euclidean_similarity_inception, f)
    print("Saved euclidean_similarity_inception to local")
    
    #-----------------------------------------------------------------------------------------
    # clustering
    #-----------------------------------------------------------------------------------------
    sub_style['db_low_euc'] = dbscanClustering(low_level_features, dist_type = 'euclidean')
    sub_style['db_low_cos'] = dbscanClustering(low_level_features, dist_type = 'cosine')
    num_clusters = min(round(len(sub_style['db_low_euc'])/20), 10)
    sub_style['ap_low_euc'] = affinityPropagation(low_level_features, dist_type = 'euclidean')
    sub_style['agg_low_euc'] = agglomerativeClustering(low_level_features, num_clusters, dist_type = 'euclidean')
    sub_style['k_low_euc'] = kMeans(low_level_features, num_clusters)
    print("Clustering using low level features completed")
    #silhouette = silhouette_score(euclidean_similarity_low, sub_style['db_low_euc'], metric='precomputed')
    #print("low - euc - db", silhouette)

    sub_style['db_vgg16_euc'] = dbscanClustering(vgg16_features, dist_type = 'euclidean', eps = 0.25)
    sub_style['db_vgg16_cos'] = dbscanClustering(vgg16_features, dist_type = 'cosine', eps = 0.25)
    num_clusters = min(round(len(sub_style['db_vgg16_euc'])/20), 10)
    sub_style['ap_vgg16_euc'] = affinityPropagation(vgg16_features, dist_type = 'euclidean')
    sub_style['agg_vgg16_euc'] = agglomerativeClustering(vgg16_features, num_clusters, dist_type = 'euclidean')
    sub_style['k_vgg16_euc'] = kMeans(vgg16_features, num_clusters)
    print("Clustering using vgg16 features completed")
    #silhouette = silhouette_score(euclidean_similarity_vgg16, sub_style['db_vgg16_euc'], metric='precomputed')
    #print("vgg16 - euc - db", silhouette)

    sub_style['db_vgg19_euc'] = dbscanClustering(vgg19_features, dist_type = 'euclidean', eps = 0.25)
    sub_style['db_vgg19_cos'] = dbscanClustering(vgg19_features, dist_type = 'cosine', eps = 0.25)
    num_clusters = min(round(len(sub_style['db_vgg19_euc'])/20), 10)
    sub_style['ap_vgg19_euc'] = affinityPropagation(vgg19_features, dist_type = 'euclidean')
    sub_style['agg_vgg19_euc'] = agglomerativeClustering(vgg19_features, num_clusters, dist_type = 'euclidean')
    sub_style['k_vgg19_euc'] = kMeans(vgg19_features, num_clusters)
    print("Clustering using vgg19 features completed")
    #silhouette = silhouette_score(euclidean_similarity_vgg19, sub_style['db_vgg19_euc'], metric='precomputed')
    #print("vgg19 - euc - db", silhouette)

    sub_style['db_resnet_euc'] = dbscanClustering(resnet_features, dist_type = 'euclidean', eps = 0.25)
    sub_style['db_resnet_cos'] = dbscanClustering(resnet_features, dist_type = 'cosine', eps = 0.25)
    num_clusters = min(round(len(sub_style['db_resnet_euc'])/20), 10)
    sub_style['ap_resnet_euc'] = affinityPropagation(resnet_features, dist_type = 'euclidean')
    sub_style['agg_resnet_euc'] = agglomerativeClustering(resnet_features, num_clusters, dist_type = 'euclidean')
    sub_style['k_resnet_euc'] = kMeans(resnet_features, num_clusters)
    print("Clustering using resnet features completed")
    #silhouette = silhouette_score(euclidean_similarity_resnet, sub_style['db_resnet_euc'], metric='precomputed')
    #print("resnet - euc - db", silhouette)

    sub_style['db_inception_euc'] = dbscanClustering(inception_features, dist_type = 'euclidean', eps = 0.25)
    sub_style['db_inception_cos'] = dbscanClustering(inception_features, dist_type = 'cosine', eps = 0.25)
    num_clusters = min(round(len(sub_style['db_inception_euc'])/20), 10)
    sub_style['ap_inception_euc'] = affinityPropagation(inception_features, dist_type = 'euclidean')
    sub_style['agg_inception_euc'] = agglomerativeClustering(inception_features, num_clusters, dist_type = 'euclidean')
    sub_style['k_inception_euc'] = kMeans(inception_features, num_clusters)
    print("Clustering using inception features completed")
    #silhouette = silhouette_score(euclidean_similarity_inception, sub_style['db_inception_euc'], metric='precomputed')
    #print("inception - euc - db", silhouette)

    results = results.append(sub_style)
    

results.to_csv('results.csv', index = False) 

