#!/opt/homebrew/bin/bash
echo "** Parameter meshgrid creation process activates"
echo "************ CAUTION : PLZ CHECK REFRESHING HYBtest PROGRAM  ******************"
python3 ./2024_08_17_Parameter_output.py -o Param

mapfile -t lines < "Param.txt"

# 각 줄을 공백으로 구분하여 배열로 변환
IFS=' ' read -r -a line1 <<< "${lines[0]}"
IFS=' ' read -r -a line2 <<< "${lines[1]}"

unset IFS

for ((i=0; i<${#line1[@]}; i++)); do
    for ((j=0; j<${#line2[@]}; j++)); do
        echo "** Passing parameters to program"
        echo "Passing values: ${line1[i]} ${line2[j]}"
        echo "${line1[i]} ${line2[j]}" | ./HYBtest

        ./build/RSJJ_NCA test.json

    done
done
