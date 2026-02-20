#!/bin/bash

git switch $1 
git rebase main 
git switch main 
git merge --ff-only $1
git push origin main
