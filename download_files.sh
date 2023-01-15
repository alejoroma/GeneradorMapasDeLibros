#!/bin/bash
path="`pwd`/pdf"
filename='book_urls.txt'
while read line; do
google-chrome -incognito &
sleep 2
xdotool type $line
sleep 2
xdotool key KP_Enter
done < $filename