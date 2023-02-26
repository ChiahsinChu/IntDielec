# add modulefiles
module load intel/17.5.239 mpi/intel/2017.5.239
module load gcc/5.5.0
module load cp2k/7.1

root_dir=$(pwd)

for dir in `ls`  
do  
    if [ -d $dir ] && [ -d $dir/ref ]
    then 
        cd $dir/ref
        echo $(pwd)
        mpiexec.hydra cp2k.popt input.inp > output.out
        touch finished_tag
        cd ../../
    fi  
done 

