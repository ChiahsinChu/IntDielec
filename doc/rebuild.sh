# clean build folder
make clean

# >>> if you change the codes >>>
## clean rst files in source folder
cd source 
rm `ls *rst | grep -v "^index.rst$"`
cd ../
sphinx-apidoc -o source ../intdielec
# <<< if you change the codes <<<

make html