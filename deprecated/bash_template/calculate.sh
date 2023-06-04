# add modulefiles
module load intel/17.5.239 mpi/intel/2017.5.239
module load gcc/5.5.0
module load cp2k/7.1


for dir in `ls`  
do  
    if [ -d $dir ] && [ -d $dir/ref ]
    then
        cd $dir
        for subdir in `ls`
        do 
            if [ -d $subdir ] && [ ! -f $subdir/finished_tag ]
            then
                cd $subdir
                # echo $(pwd)
                if [ ! -f cp2k-RESTART.wfn ]
                then
                    ln -s ../ref/cp2k-RESTART.wfn ./cp2k-RESTART.wfn
                fi
                mpiexec.hydra cp2k.popt input.inp > output.out
                touch finished_tag
                cd ../
            fi
        done  
        cd ../
    fi  
done 
