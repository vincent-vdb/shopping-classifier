#!/bin/sh

for d in `find $1 -mindepth 1 -type d`
do
	N=`find $d -mindepth 1 -type f | wc -l`
	echo "$d\t$N"
done
