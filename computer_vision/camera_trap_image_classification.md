# Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning

- [Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning](#automatically-identifying-counting-and-describing-wild-animals-in-camera-trap-images-with-deep-learning)
  - [Problem](#problem)
    - [Snapshot Serengeti (SS) Project](#snapshot-serengeti-ss-project)
  - [Current Work](#current-work)
  - [Datasets](#datasets)
  - [Architectures](#architectures)
  - [Tasks](#tasks)
      - [Task 1 (Binary Classification: Empty vs. Non-Empty)](#task-1-binary-classification-empty-vs-non-empty)
      - [Task 2 (Multiclass Classficiation: Identifying Animals)](#task-2-multiclass-classficiation-identifying-animals)
      - [~~Task 3 (Counting Animals)~~](#stask-3-counting-animalss)
      - [~~Task 4 (Additional Attributes)~~](#stask-4-additional-attributess)
  - [Experiments](#experiments)


## Problem

To better understand the complexities of natural ecosystems and better manage and protect thenm it would be helpful to have detailed, large-scale knowledge about the number, location, and behaviors of animals in the natural ecosystems. 

Currently, we use motion sensor cameras in natural habitats to assist with this task. These cameras take millions of images but the task of labelling these images, categorizing animals remain a hard task. Human volunteers have to manually go through each image and classify it as an animal or empty. This is very time-consuming and results in errors. 

In the **Snapshot Serengeti (SS)** project where a huge volunteer force was harnessed to label these images, the human volutneer specials and count labels are estimated to be 96.6% and 90.0% accuracte respectively.

### Snapshot Serengeti (SS) Project

The world's largest camera-trap project published to date, SS has 225 camera traps running continuously in *Serengeti National Park, Tanzania*. 

For this project, 28,000 registered and 40,000 unregistered volunteer have labeled 1.2 million SS capture events. 

For each image set, multiple useers label the species, number of individuals, various behaviours, and the presence of young.

## Current Work

For the current paper, the main focus was on the capture events that contain **only one species**. All the other capture events and images were removed from the dataset. This was around 1.2% per the events. 

It was found that **75%** of the capture events were classified as empty for animals. Also, the volunteers labelled the entire capture event rather than individual images. The main experiments for this research identify individual images rather than the entire event.

## Datasets

The dataset used for this research is the [Snapshot Serengeti Dataset](http://lila.science/datasets/snapshot-serengeti). This dataset contained 1.4 million images with a total of 301,400 capture events *(set of images in a single motion capture trap)*

The dataset consists of 48 different species spread across the Serengeti National Park. 

In this dataset, 25% of images are labelled as non-empty *(containing animals)* by the human-volunteers. 

To overcome the issue of overfitting where the model just memorizes the examples in the training and testing set, the entire capture event, containing 3 images each, was put in either training or test set. 

Out of the 301,400 capture events 284,400 random capture events were put in the training set and rest in the testing set. There was another testing set that was labelled by the expert scientist and was used as a gold-standard in testing for all the models. This testing set contained 3,800 capture events.

## Architectures

Nine different architectures were tested to find the highest-performing networks. Each model was only trained one time because it is computationally expensive. Also, it is theoretically and empirically suggested that different Deep Neural Networks trained with teh same architecture but initialized differently, often converge to similar performance levels. 

The following architectures were used:
<pre>
- AlexNet:    (Num layers 8)
- VGG:        (Num layers 22)
- NiN:        (Num layers 16)
- GoogLeNet:  (Num layers 32)
- ResNet:     (Num layers 18)
- ResNet:     (Num layers 34)
- ResNet:     (Num layers 50)
- ResNet:     (Num layers 101)
- ResNet:     (Num layers 152)
</pre>

## Tasks

This research is divided into multiple tasks and each task has a separate goal that was later combined to automate the camera-trap image labelling task for the Serengeti Dataset.

#### Task 1 (Binary Classification: Empty vs. Non-Empty)
In this task, a single model was trained to detect if the image contains an animal or not. Since it was initially noted that 75% of the images were empty, automating this task meant that the human labor was reduced by 75%. 

#### Task 2 (Multiclass Classficiation: Identifying Animals)
In this task, multiple single models and an ensemble of models was trained to classify images as one of the 48 different species in the Serengeti Dataset. 

#### ~~Task 3 (Counting Animals)~~

> **Not in the current scope**

#### ~~Task 4 (Additional Attributes)~~
> **Not in the current scope**

## Experiments

Given the tasks above, the experiments showed that a **two-staged pipeline** outperformed a **one-step** pipeline.

The first stage was solving the binary classification task. This was the Task 1. All the 25% of non-empty images were taken. These were 757,000 images and from the 75% of empty images, the same amount of images were selected randomly. 

The training set for this task contained 1.4 million images and the test set contained 105,000 images. 

> SS dataset contains labels for capture events so all the images in a single capture event were assigned the same label. 

**VGG a

In the second stage, most of the information-extraction was done. This stage included three tasks, namely task 2, task 3 and task 4. One model was trained to simultaneously perform all of these three tasks. This technique is popularly called *Multi-task Learning*. 

The reason for doing this, was since these three tasks are related. 