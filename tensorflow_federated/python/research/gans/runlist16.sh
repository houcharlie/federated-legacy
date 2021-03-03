#!/bin/bash
# script to generate a list of submit files and submit them to condor
EXEC=$1
runlist=$2
jobname=$3



# set up results directory
dir=$PWD/$jobname/runlist_`date '+%y%m%d_%H.%M.%S'`
echo "Setting up results directory: $dir"
mkdir $PWD/$jobname
mkdir $dir
# preamble

i=0

while read p; do
  if [ $i -lt 1 ]
  then
    i=$((i+1))
    continue
  fi
  echo "$EXEC $p"
  echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-AI
#SBATCH --gres=gpu:volta16:1
#SBATCH -t 48:00:00
#SBATCH -C EGRESS
#SBATCH --mail-type=ALL
#SBATCH -o $dir/output_$i.out
source /etc/profile.d/modules.sh
module load cuda/10.1
export PATH=$HOME/bin:$PATH
source /home/houc/venv/bin/activate
sh $EXEC $p" >> $dir/runlist_$i.job
  sbatch $dir/runlist_$i.job
  i=$((i+1))
done <$runlist
#submit to condor
