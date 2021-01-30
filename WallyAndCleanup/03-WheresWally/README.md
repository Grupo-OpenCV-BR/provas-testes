# Description

At the bottom of the sea? At the peak of the tallest mountain? Where might that Wally chap be? Is there a way to automatically determine Where's Wally?

A number of images with random backgrounds is provided. In each image, the same Wally picture was placed at a random position, with random rotation and a small perspective transform.
The following items are provided as `TrainSet` dataset:

- A collection of random images with Wally's picture at a random position
- The annotation json files (LabelMe standard), determining where Wally's picture is in each of the train images

The following items are provided as `TestSet` dataset:

- A collection of random images with Wally's picture at a random position

The following items are provided as `ReferenceData`:

- The original Wally picture
- A csv file containing the centroid of Wally's picture in each of the train images

# Objective

The objective of this test is to find a way to automatically detect Wally's picture in each of the `TestSet` images, giving the centroid of the picture in each image as the final answer.

# Important details

- Wally does not like being found. When asked about appearing in our test, he asked to personaly review the data first. We wonder if he tampered with the data to disturb our solutions...
- The dataset was split in order to have unseen data for test analysis. We took 20% of the total data (randomly)
- The annotations are in LabelMe standard. You can find the software [here](https://github.com/wkentaro/labelme)
- The CSV file contains the filename in the first column, the `x` position of the centroid in the second column and the `y` position of the centroid in the third column
- This test does not require a defined image processing algorithm to be used. The candidate is free to choose any kind of image processing pipeline to reach the best answer
- Depending on the chosen approach, not all provided files might be needed. We provide different resources so that different approaches are possible, but the candidate should feel free to use or discard any of the provided resources
- Replicate the data format for submission, i.e. the answer must be provided as a CSV file with the filename in the first column, the `x` coordinate for the centroid of the picture in the second column and the `y` coordinate for the centroid of the picture in the third column, similar to what is provided in the `TrainSet` dataset
