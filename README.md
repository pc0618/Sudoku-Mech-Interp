# Sudoku-Mech-Interp

This code was part of a CS 229 Machine learnign final project at Stanford which can be found here https://drive.google.com/file/d/18AwCtOgmxSEacR4ekNfE3f8Xfx1slu2D/view?usp=drive_link

While neural networks have achieved superhuman performance in several domains including
games such as Chess, Atari, and Go, understanding what these models learn and how they represent these games internally remains a significant
challenge. Recent work in the field of mechanistic interpretability has proposed new methods of performing weight-based interpretability for transformers that utilize bilinear MLPs. 
In this work we train a modified GPT-2 based transformer with
bilinear MLPs to solve Sudoku puzzles. Through
techniques based on linear probing and various
matrix decompositions of the MLP layers of our
model, we find preliminary evidence that suggests
that our model may learn Sudoku strategies akin
to what human players routinely use.

