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
        if [ -f $req_file ]
        then
            echo "Requirement file found."
            echo 'Creating the virtual env' \"$1\"
            python3 -m venv $1
            . $1/bin/activate
            echo "Installing the requirements ..."
            #pip install -r $req_file
            echo "Requirement installed."

            echo "Read the read me file to understand how to run the project"
            echo "https://github.com/I-Akrout/Predict-customer-churn-clean-code/blob/main/README.md"
        else
            echo "requirement file not found."
            echo "Download the requirement file from this repo."
            echo "https://github.com/I-Akrout/Predict-customer-churn-clean-code/blob/main/create_env.sh"
        fi
        python3 -m venv $1
    fi
fi