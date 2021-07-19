
# CSCS-USI HPC Summer School 2021

This repository contains the materials used in the Summer School, including source code, lecture notes and slides.
Material will be added to the repository throughout the course, which will require that students either update their copy of the repository, or download/checkout a new copy of the repository.

## Announcements

We will be using Slack to post news and links relevant to the event: you should have received an invitation to join the Summer School Slack workspace.

## Schedule (updated 19.07.2021)

[Full PDF](https://github.com/eth-cscs/SummerSchool2021/files/6840297/2021.-.Effective.High-Performance.Computing.Data.Analytics.Summer.School.3.pdf)

![Schedule](https://user-images.githubusercontent.com/4578156/126151194-f6d146c7-c607-4c62-9890-72d3851d84af.png)

## Obtaining a copy of this repository

### On your own computer

You will want to download the repository to your laptop to get all of the slides.
The best method is to use git, so that you can update the repository as more slides and material are added over the course of the school.
So, if you have git installed, use the same method as for Piz Daint below (in a directory of your choosing).

You can also download the code as a zip file by clicking on the green __Clone or download__ button on the top right hand side of the github page, then clicking on __Download zip__.

### On Piz Daint

Piz Daint has git installed, so we can download the code directly to where we will be working on the exercises and practicals.

```bash
# log onto Piz Daint ...

# go to scratch
cd $SCRATCH

# it is a good policy to put it in a personal path
mkdir johnsmith
cd johnsmith
git clone https://github.com/eth-cscs/SummerSchool2021.git
```

## Updating the repository

Lecture slides and source code will be continually added and updated throughout the course.
To update the repository you can simply go inside the path

```
git pull origin master
```

There is a posibility that you might have a conflict between your working version of the repository and the origin.
In this case you can ask one of the assistants for help.

# How to access Piz Daint

This will be covered in the lectures and you can find more details in the [CSCS User Portal](https://user.cscs.ch/access/running/piz_daint/).
