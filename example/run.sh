C=`nproc`
line=`lscpu | grep "Thread(s) per core"`
index=`echo $line | wc -w`
T=`echo $line |awk -v i=$index '{print $i}'`

N=`expr $C / $T`
echo " "
echo "<<<<<<<  Simulation of VGG-16 CONVOLUTION LAYERS: Input-Image = 226x226x3  Batch=64  >>>>>>>" 
OMP_PROC_BIND=spread,close OMP_PLACES=cores MKL_DYNAMIC=false OMP_NESTED=1 OMP_NUM_THREADS=$N,2 KMP_HOT_TEAMS_MODE=1 KMP_HOT_TEAMS_MAX_LEVEL=2 OMP_MAX_ACTIVE_LEVELS=1 ./vgg_winograd  64 $1

