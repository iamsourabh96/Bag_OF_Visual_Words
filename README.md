# Bag_OF_Visual_Words

The general idea of bag of visual words (BOVW) is to represent an image as a set of features, thus enabling compact representation of the image.   
Visual words representation can be useful for place recognition task, loop closure detection, finding similar images, etc. Visual word representations of images are also somewhat robust to orientation and illumination changes, depending on the descriptor used.    
This implementation uses SIFT features for computing the bag of words (codebook) due its robustness to enviornmental and orientation changes.

## Building a Bag Of Visual Words Dictionary using k-means (codebook)

Feature desciptors are extracted from all images in the dataset using SIFT and all these feature vectors are then condensed to a compact representaion using k-means clustering to form the codebook.

![Screenshot from 2020-11-17 20-00-28](https://user-images.githubusercontent.com/49958651/99468889-a71ee480-290f-11eb-9c5c-9a19f758ee82.png)

## Computing Histograms for each image

Once the codebook is computed and fixed, each image can then be transformed into a histogram representaion - counting the number of occurances of the features in codebook observed in image. 

![hist](https://user-images.githubusercontent.com/49958651/99470769-4abdc400-2913-11eb-95cc-eafc93b9f71a.jpeg)


## Finding Similar Images

Given a query image - transfrom it into histogram representaion and then compare it with pre-computed histograms of all images (histbook) in the dataset and return K best matches.

## Result

#### Query Image

![000000](https://user-images.githubusercontent.com/49958651/99471365-637aa980-2914-11eb-8a4a-e2a5ebc27a28.png)


#### Loop Closure Detected Image

![001044](https://user-images.githubusercontent.com/49958651/99471449-8d33d080-2914-11eb-8f58-d80b04324ab8.png)

