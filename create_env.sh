# Using this bash file we create a virtual envirement to run the churn library
# and test it
#
# 
# Author: Ismail Akrout
#
# Date: October 2nd, 2021
#
# Version: 0.0.1

if [ $# -eq 0 ]
then
    echo "Please provide the virtual env name."
else
    pattern=" |'"
    if [[ $1 =~ $pattern ]]
    then
        echo 'Virtual env name should not contain a spacing characters'
    else
        req_file="./requirements.txt"
        if [ -f "" ]
        echo 'Creating the virtual env' \"$1\"
        python3 -m venv $1
    fi
fi