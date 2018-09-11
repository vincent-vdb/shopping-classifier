#!/bin/sh

# Alexandre Coninx
# ISIR CNRS/UPMC
# 21/07/2016


for f in *.MOV
do
	DIR=`echo $f | sed "s/\.[^\.]*$//"`
	mkdir -p $DIR
	ffmpeg -i $f $DIR/image-%04d.jpg > /dev/null
	echo "$f done"
done
