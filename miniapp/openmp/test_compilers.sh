current_env() {
    env=`module list |& grep PrgEnv |& awk '{print $2}'`
}

change_env() {
    current_env
    module swap $env PrgEnv-$1
}

target=gpu
target=mc

module load daint-$target

dim=256
steps=200

change_env cray
make clean main > cmp
cp main main.cray
srun -C$target main.cray $dim $dim $steps 0.01 &> tmp
echo ============= Cray ===============
grep second tmp

change_env gnu
module swap gcc/7.3.0
make clean main > cmp
cp main main.gnu
srun -C$target main.gnu $dim $dim $steps 0.01 &> tmp
echo ============= GNU ===============
grep second tmp

change_env intel
make clean main > cmp
cp main main.intel
srun main.intel $dim $dim $steps 0.01 &> tmp
echo ============= Intel ===============
grep second tmp

