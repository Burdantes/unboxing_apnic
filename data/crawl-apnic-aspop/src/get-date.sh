#!/usr/bin/env bash


cd $(dirname $0)

export DATA_DIR=../data



TS="date -Iseconds"

function usage() {
  echo "Usage: ${0} [date]"
  echo ""
  echo "    date: datestamp formatted as YYYY-MM-DD, uses (today - 2 days) if not provided (optional)"
  echo ""
}

echo "$(${TS}) starting data pull on $(hostname -f)"

# check if data directory exists
if [ -d "$DATA_DIR" ]; then
    # $DATA_DIR exists and is a directory (or a symlink to one), cd into it
    cd $DATA_DIR
elif [ ! -e "$DATA_DIR" ]; then
    # nothing named $DATA_DIR exists, mkdir then cd into it
    mkdir $DATA_DIR
    cd $DATA_DIR
else
    # something that isn't a directory named "$DATA_DIR" exists, exit
    echo "$(${TS}) ${DATA_DIR} exists and isn't a directory, exiting..."
    exit 1
fi

# do we have an argument?
input_dateToGet=${1:-0}

if [[ ${input_dateToGet} == 0 ]]; then
    # set to today's date - 2
    # because we can never really get today's date
    dateToGet=$(date --date="-2 days" -I)
    echo "$(${TS}) no input date (YYYY-MM-DD) provided, setting date := ${dateToGet}"
else
    # try to parse the date with `date`
    dateToGet=$(date --date=${input_dateToGet} -I 2>/dev/null)
    retVal=$?

    if [[ ${retVal} != 0 ]]; then
        echo "$(${TS}) error parsing date, exiting..."
        echo ""
        usage
        exit 1
    fi
fi

echo "$(${TS}) getting data on date := ${dateToGet}"

# split our string into YEAR-MONTH-DATE
declare -a dateArray=( )
readarray -d "-" -t dateArray <<< ${dateToGet}

# strip newlines at the end
YEAR=${dateArray[0]//[$'\t\r\n']}
MONTH=${dateArray[1]//[$'\t\r\n']}
DAY=${dateArray[2]//[$'\t\r\n']}

# build URL
#   - w=120             120 days (this doesn't seem to change anything)
#   - d=DD/MM/YYYY      note date format
#   - f=c               c := csv, j := json
URL="https://stats.labs.apnic.net/cgi-bin/aspop?w=120&d=${DAY}/${MONTH}/${YEAR}&f=c"

declare -a OPTS=()
OPTS+=(--compressed)
OPTS+=(-L)
OPTS+=(${URL})

# download URL
#
# TODO probably keep a log as to when exactly we downloaded these files
OUTPUT_FN=aspop.${dateToGet}.csv

# if the destination exists, exit
# TODO support -f flag to overwrite files
if [ -f ${OUTPUT_FN} ]; then
    echo "$(${TS}) output file ${OUTPUT_FN} already exists, pass '-f' to overwrite, exiting..."
    exit 1
fi

echo "$(${TS}) downloading ${URL} to ${OUTPUT_FN}"
echo "${OUTPUT_FN} ${URL} $(date -Iseconds)"

curl "${OPTS[@]}" > ${DATA_DIR}/${OUTPUT_FN}
