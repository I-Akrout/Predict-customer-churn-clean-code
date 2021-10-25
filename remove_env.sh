# Using this bash file we remove the virtual envirement used to run the churn library
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
        if [ -d $1 ]
        then
            echo "Deleting the virtual env" \"$1\"
            rm -rf $1
        else
            echo "Env folder" \"$1\" "not found"
        fi
    fi
fi