# A/B testing

This repo contains information about how to conduct A/B tests. All info and code is in the jupyter notebooks. The whole process is divided into 3 parts:
* Preparation
* A/A test
* A/B test

## Preparations part (1 part)
The preparation part contains information, how to split the data into groups, that will be using in experiment later. Also we'll plot choosen distribution metric to see, if our data in first group looks similar to data in second group.

## A/A test (2 part)
When we have splitted our users into groups, we may want to know, if this data in different groups are identical right now, before the experiment. To do this, we conduct A/A test. The A/A test allows to find out, if the distributions from 2 groups doesn't have statistical significances. Because if the data from 2 groups already different, so what's the point of experiment?

## A/B test (3 part)
After we know, that the data in different groups doesn't have any statistical significances, we can conduct A/B test. This part contains different approaches to do A/B testing, like testing standard CTR metric, smoothed CTR, linearized CTR and others.