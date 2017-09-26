#!/bin/bash

abspath="/home/taylanbil/taylanbil.github.io/_posts"
ipynbfn=`basename $1`
mdfn=$abspath/`date +%Y-%m-%d`-`echo $ipynbfn | sed s/ipynb$/md/`

postdate=`date +"%Y-%m-%d %H:%M:%S"`
echo "---
title: 
date: $postdate
categories: 
permalink: 
layout: taylandefault
published: false
---

" > $mdfn

jupyter-nbconvert --to markdown $1 --stdout >> $mdfn
echo "post $mdfn is created. MAKE SURE TO EDIT THE TITLE, CATEGORIES and PERMALINK"

